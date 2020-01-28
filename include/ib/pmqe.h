#pragma once
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include "util/functional.h"
#include "util/iterators.h"
#include "util/ranges.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "fd/grid.h"
#include "types.h"
#include "delta.h"
#include "roma.h"
#include "cosine.h"
#include "indexing.h"
#include "sweep.h"
#include "interpolate.h"

namespace ib {
namespace pmqe {

template <std::size_t meshwidths>
struct index {
private:
	int inner_index;
	int inner_weight;
	int outer_index;
	int outer_weight;

	constexpr int value() const { return inner_index * outer_weight + outer_index; }
public:
	constexpr index&
	operator*=(const index& o)
	{
		inner_index  += inner_weight * o.inner_index;
		outer_index  += outer_weight * o.outer_index;
		inner_weight *= o.inner_weight;
		outer_weight *= o.outer_weight;
		return *this;
	}

	constexpr index(int index, int lower, int upper) :
		inner_index((index - lower) % meshwidths),
		inner_weight(meshwidths),
		outer_index((index - lower) / meshwidths),
		outer_weight((upper - lower + meshwidths-1) / meshwidths) {}

	explicit constexpr operator int() const { return value(); }
};

template <std::size_t meshwidths>
constexpr auto
combine(index<meshwidths> l, const index<meshwidths>& r)
{
	l *= r;
	return l;
}

template <typename grid_type, typename delta_type>
struct sorter : indexing::sorter<grid_type> {
private:
	using base = indexing::sorter<grid_type>;
	using traits = delta::traits<delta_type>;
	static constexpr auto meshwidths = traits::meshwidths;
	using base::dimensions;

	static constexpr auto
	compute(const grid_type& grid)
	{
		using namespace util::functional;
		auto p = [] (auto ... v) { return (v * ... * 1); };
		auto k = [] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto weight = comp.points() + solid;
			return (weight + meshwidths - 1) / meshwidths;
		};
		return apply(p, map(k, grid.components()));
	}
public:
	constexpr auto
	decompose(int index) const
	{
		using namespace util::functional;
		auto outer_index = index % blocks;
		auto inner_index = index / blocks;

		auto k = [&] (const auto& comp) -> int
		{
			auto solid = comp.solid_boundary;
			auto weight = comp.points() + solid;
			auto blocks = (weight + meshwidths - 1) / meshwidths;
			auto outer = outer_index % blocks;
			outer_index /= blocks;
			auto inner = inner_index % meshwidths;
			inner_index /= meshwidths;
			return meshwidths * outer + inner - solid;
		};
		return base::decompose(k);
	}

	using sort_index_type = index<meshwidths>;
	constexpr sorter(const grid_type& g, const delta_type&) :
		base(g), blocks(compute(g)) {}

	const int blocks;
};

template <typename grid_tag, typename domain_type>
struct spread {
private:
	using cuda_tag = thrust::system::cuda::tag;
	using policy = thrust::device_execution_policy<cuda_tag>;
	using delta_type = delta::cosine;
	using traits = delta::traits<delta_type>;
	static constexpr policy exec;
	static constexpr auto dimensions = domain_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto values = detail::cpow(meshwidths, dimensions);
	static constexpr delta_type phi;
	using point = ib::point<dimensions>;

	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp) { return fd::grid{tag, domain, comp}; };
		return map(k, fd::components(domain));
	}

	static auto
	heads(int n, const util::memory<int>& indices)
	{
		util::memory<int> buffer(n);
		auto* idata = indices.data();
		auto* bdata = buffer.data();
		auto* bend = thrust::unique_copy(exec, idata, idata+n, bdata);
		auto m = bend - bdata;

		util::memory<int> heads(m);
		auto* jdata = heads.data();
		thrust::lower_bound(exec, idata, idata+n, bdata, bend, jdata);
		return std::make_pair(std::move(heads), std::move(buffer));
	}

	template <typename grid_type>
	static auto
	index(int n, const grid_type& grid, const matrix& x)
	{
		util::memory<int> indices(n);
		util::memory<int> permutation(n);
		auto* xdata = x.values();
		auto* idata = indices.data();
		auto* jdata = permutation.data();

		ib::sweep sweep{0, values, phi, sorter{grid, phi}};
		auto k = [=] __device__ (int tid)
		{
			point z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto j = sweep.sort(z);
			idata[tid] = j;
			jdata[tid] = tid;
		};
		util::transform<128, 7>(k, n);

		thrust::sort_by_key(exec, idata, idata+n, jdata);
		return std::pair{std::move(indices), std::move(permutation)};
	}

	template <typename grid_type>
	static auto
	reduce(int n, int m, const util::memory<int>& uniques,
			const util::memory<int>& heads, const grid_type& grid,
			const matrix& x, const double* fdata,
			const util::memory<int>& indices, const util::memory<int>& permutation)
	{
		using namespace util::functional;
		using util::ranges::enumerate;
		constexpr auto v = [] (auto&& ... c) { return (c.resolution() * ... * 1); };
		auto res = apply(v, grid.components());
		vector output(fd::size(grid), linalg::zero);

		sorter idx{grid, phi};
		ib::sweep sweep{0, values, phi, idx};
		auto blocks = idx.blocks;

		auto* idata = indices.data();
		auto* jdata = permutation.data();
		auto* hdata = heads.data();
		auto* qdata = uniques.data();
		auto* xdata = x.values();

		auto* odata = output.values();
		auto k = [=] __device__ (int tid, int start)
		{
			double v[values] = {0.0};
			point z;

			auto offset = start + tid;
			auto head = hdata[offset];
			auto tail = offset+1 < m ? hdata[offset+1] : n;
			for (int k = head; k < tail; ++k) {
				auto j = jdata[k];
				for (int i = 0; i < dimensions; ++i)
					z[i] = xdata[n * i + j];
				double f = fdata[j];
				auto w = sweep.values(z);
				for (auto [i, w]: w | enumerate)
					v[i] += w * f;
			}

			auto j = sweep.indices(idata[head]);
			for (auto [i, j]: j | enumerate)
				if (j >= 0) odata[j] += res * v[i];
		};

		auto* qnow = qdata;
		for (int i = 0; i < values; ++i) {
			auto* qend = thrust::lower_bound(exec, qnow, qdata+m, (i + 1) * blocks);
			auto threads = qend - qnow;
			auto start = qnow - qdata;
			util::transform(k, threads, start);
			qnow = qend;
		}

		return output;
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(),
	                                      std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr spread(const grid_tag& tag, const domain_type& domain) :
		grids(construct(tag, domain)) {}

	auto
	operator()(int n, const matrix& x, const matrix& f) const
	{
		timer s{"spread"};
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;
		auto* fdata = f.values();
		auto k = [&] (const auto& grid, auto m)
		{
			static constexpr auto i = decltype(m)::value;
			auto [indices, permutation] = index(n, grid, x);
			auto [pointers, compressed] = heads(n, indices);
			auto* f = &fdata[n * i];
			return reduce(n, pointers.size(), compressed,
					pointers, grid, x, f, indices, permutation);
		};
		return map(k, grids, sequence{});
	}
};

using ib::interpolate;

} // namespace pmqe
} // namespace ib
