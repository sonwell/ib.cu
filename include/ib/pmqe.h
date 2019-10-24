#pragma once
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "fd/grid.h"
#include "novel.h"
#include "sweep.h"
#include "types.h"

namespace ib {
namespace pmqe {

struct index {
	int inner_index;
	int inner_weight;
	int outer_index;
	int outer_weight;

	constexpr index&
	operator*=(const index& idx)
	{
		inner_index += inner_weight * idx.inner_index;
		inner_weight *= idx.inner_weight;
		outer_index += outer_weight * idx.outer_index;
		outer_weight *= idx.outer_weight;
		return *this;
	}

	constexpr int value() const { return inner_index * outer_weight + outer_index; }
	constexpr operator int() const { return value(); }

	constexpr index(int index, int lower, int upper) :
		inner_index((index - lower) % 4),
		inner_weight(4),
		outer_index((index - lower) / 4),
		outer_weight((upper - lower + 3) / 4) {}
};

constexpr index
operator*(index l, const index& r)
{
	l *= r;
	return l;
}

template <typename grid_type>
struct indexer {
public:
	using point_type = typename grid_type::point_type;
	using units_type = typename grid_type::units_type;
	using indices_type = typename grid_type::indices_type;
private:

	static constexpr decltype(auto)
	block_info(const grid_type& grid)
	{
		using namespace util::functional;
		auto k = [] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			return (comp.points() + solid + 3) / 4;
		};
		auto counts = map(k, grid.components());
		auto total = apply(partial(foldl, std::multiplies<void>{}), counts);
		return std::make_pair(counts, total);
	}

	using block_info_t = decltype(block_info(std::declval<grid_type>()));

	const grid_type& _grid;
	typename block_info_t::first_type _blocks;
	typename block_info_t::second_type _total;


	constexpr decltype(auto)
	components() const
	{
		return _grid.components();
	}

	template <typename point_type, typename indexing_type>
	constexpr auto
	build(const point_type& p, indexing_type&& indexing) const
	{
		using namespace util::functional;
		constexpr std::multiplies<void> op = {};

		auto indices = map(indexing, p, components());
		return apply(partial(foldl, op), indices);
	}

	constexpr indexer(const grid_type& grid, const block_info_t& info) :
		_grid(grid), _blocks(info.first), _total(info.second) {}
public:
	constexpr auto
	decompose(int index) const
	{
		using namespace util::functional;
		indices_type indices = {0};
		auto outer_index = index % _total;
		auto inner_index = index / _total;

		auto k = [&] (int& i, int blocks, const auto& comp)
		{
			auto solid = comp.solid_boundary;
			i = 4 * (outer_index % blocks) + inner_index % 4 - solid;
			outer_index /= blocks;
			inner_index /= 4;
		};
		map(k, indices, _blocks, components());
		return indices;
	}

	constexpr auto
	sort(const units_type& u) const
	{
		auto k = [] (double v, const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto shift = comp.shift();
			auto i = (int) util::math::floor(v - shift);
			return index{i, -solid, comp.points()};
		};
		return build(u, k);
	}

	constexpr auto
	grid(const indices_type& u) const
	{
		auto k = [] (int i, const auto& comp)
		{
			auto j = comp.index(i);
			return novel::index{j, 0, comp.points()};
		};
		return build(u, k);
	}

	constexpr auto blocks() const { return _total; }

	constexpr indexer(const grid_type& grid) :
		indexer(grid, block_info(grid)) {}
};

using fd::__1::delta;
using fd::__1::shift;

template <typename grid_type>
struct sweep {
	static constexpr auto dimensions = grid_type::dimensions;
	static constexpr auto total_values = 1 << (2 * dimensions);
	static constexpr cosine_delta phi = {};
	using container_type = std::array<double, total_values>;

	const grid_type& grid;
	int _blocks;

	static constexpr auto mask(int i, int n) { return (i & (3 << (2 * n))) >> (2 * n); };

	constexpr auto
	values(const delta<dimensions>& dx, double f) const
	{
		std::array<double, total_values> v{0.0};
		double weights[dimensions][4];
		for (int i = 0; i < dimensions; ++i) {
			auto v0 = phi(dx[i] - 1);
			auto v1 = phi(dx[i] + 0);
			weights[i][0] = v0;
			weights[i][1] = v1;
			weights[i][2] = 0.5 - v0;
			weights[i][3] = 0.5 - v1;
		}

		for (int j = 0; j < total_values; ++j) {
			double w = f;
			for (int i = 0; i < dimensions; ++i)
				w *= weights[i][mask(j, i)];
			v[j] = w;
		}

		return v;
	}

	constexpr auto
	indices(int index) const
	{
		std::array<int, total_values> values = {0};
		indexer idx{grid};

		auto indices = idx.decompose(index);
		for (int i = 0; i < total_values; ++i) {
			shift<dimensions> s = {0};
			for (int j = 0; j < dimensions; ++j)
				s[j] = mask(i, j) - 1;
			values[i] = idx.grid(indices + s);
		}

		return values;
	}

	constexpr int blocks() const { return _blocks; }

	constexpr sweep(const grid_type& grid) :
		grid(grid), _blocks(indexer<grid_type>{grid}.blocks()) {}
};

template <typename grid_tag, typename domain_type>
struct spread {
public:
	static constexpr auto dimensions = domain_type::dimensions;
	static constexpr auto values = 1 << (2 * dimensions);
//private:
	static constexpr thrust::device_execution_policy<thrust::system::cuda::tag> exec = {};

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
		using point_type = typename grid_type::point_type;
		util::memory<int> indices(n);
		util::memory<int> permutation(n);
		auto* xdata = x.values();
		auto* idata = indices.data();
		auto* jdata = permutation.data();

		auto k = [=] __device__(int tid)
		{
			point_type z;
			indexer idx{grid};
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto u = grid.units(z);
			auto j = idx.sort(u);
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
			const util::memory<int>& heads, const grid_type& grid, const matrix& x, const double* fdata,
			const util::memory<int>& indices, const util::memory<int>& permutation)
	{
		using namespace util::functional;
		using point_type = typename grid_type::point_type;
		using sweep_type = sweep<grid_type>;
		using container_type = typename sweep_type::container_type;
		constexpr auto gr = [] (const auto& c) { return c.resolution(); };
		constexpr auto prod = partial(foldl, std::multiplies<double>{}, 1.0);
		auto res = apply(prod, map(gr, grid.components()));
		auto blocksize = sweep_type{grid}.blocks();
		auto size = fd::size(grid);
		vector output(size, linalg::zero);

		auto* idata = indices.data();
		auto* jdata = permutation.data();
		auto* hdata = heads.data();
		auto* qdata = uniques.data();
		auto* xdata = x.values();

		auto* odata = output.values();
		auto k = [=] __device__ (int tid, int start)
		{
			container_type v{0.0};
			sweep_type sweep(grid);
			point_type z;

			auto head = hdata[start + tid];
			auto tail = tid + 1 < m ? hdata[start + tid+1] : n;
			for (int k = head; k < tail; ++k) {
				auto j = jdata[k];
				for (int i = 0; i < dimensions; ++i)
					z[i] = xdata[n * i + j];
				double f = fdata[j];
				auto u = grid.units(z);
				auto d = grid.difference(u);
				auto w = sweep.values(d, f);
				for (int i = 0; i < values; ++i)
					v[i] += w[i];
			}

			auto j = sweep.indices(idata[head]);
			for (int i = 0; i < values; ++i)
				odata[j[i]] = res * v[i];
		};

		auto* qnow = qdata;
		for (int i = 0; i < values; ++i) {
			auto* qend = thrust::lower_bound(exec, qnow, qdata+m, (i + 1) * blocksize);
			auto threads = qend - qnow;
			auto start = qnow - qdata;
			util::transform(k, threads, start);
			cuda::throw_if_error(cudaGetLastError());
			qnow = qend;
		}

		return output;
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(), std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr spread(const grid_tag& tag, const domain_type& domain) :
		grids(construct(tag, domain)) {}

	auto
	operator()(int n, const matrix& x, const matrix& f) const
	{
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

} // namespace pmqe
} // namespace ib
