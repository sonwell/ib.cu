#pragma once
#include <thrust/execution_policy.h>
#include "util/functional.h"
#include "util/iterators.h"
#include "util/ranges.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "fd/grid.h"
#include "types.h"
#include "sweep.h"
#include "delta.h"
#include "roma.h"
#include "cosine.h"
#include "interpolate.h"

namespace ib {
namespace novel {

template <typename grid_tag, typename domain_type, typename delta_type>
struct spread {
private:
	using cuda_tag = thrust::system::cuda::tag;
	using policy = thrust::device_execution_policy<cuda_tag>;
	using traits = delta::traits<delta_type>;
	static constexpr policy exec;
	static constexpr auto dimensions = domain_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto values = detail::cpow(meshwidths, dimensions);
	static constexpr delta_type phi;
	using point = ib::point<dimensions>;

	static constexpr auto
	largest_divisor_under(int n)
	{
		if (n >= values) return values;
		for (int i = n; i > 0; --i)
			if (values % i == 0)
				return i;
		return 1;
	}

	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp) { return fd::grid{tag, domain, comp}; };
		return map(k, fd::components(domain));
	}

	static auto
	uniques(int n, const util::memory<int>& indices)
	{
		util::memory<int> buffer(n);
		auto* idata = indices.data();
		auto* bdata = buffer.data();
		return thrust::unique_copy(exec, idata, idata+n, bdata) - bdata;
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

		indexing::sorter idx{grid};
		auto k = [=] __device__ (int tid)
		{
			point z;
			ib::sweep sweep{0, values, phi, idx};
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
	reduce(int n, int q, const grid_type& grid, const matrix& x, const double* fdata,
			const util::memory<int>& indices, const util::memory<int>& permutation)
	{
		using namespace util::functional;
		using namespace util::math;
		using util::ranges::enumerate;
		constexpr auto per_sweep = largest_divisor_under(10);
		using container_type = values_container<per_sweep>;
		constexpr auto sweeps = (values + per_sweep - 1) / per_sweep;
		constexpr auto gr = [] (auto&& ... c) { return (c.resolution() * ... * 1); };

		auto res = apply(gr, grid.components());
		auto size = fd::size(grid);
		util::memory<container_type> values(n);
		util::memory<container_type> reduced_values(q);
		util::memory<int> reduced_keys(q);

		auto* vdata = values.data();
		auto* wdata = reduced_values.data();
		auto* kdata = reduced_keys.data();
		auto* idata = indices.data();
		auto* jdata = permutation.data();
		auto* xdata = x.values();

		vector outputs[per_sweep];
		double* odata[per_sweep];
		for (int i = 0; i < per_sweep; ++i) {
			outputs[i] = vector(size, linalg::zero);
			odata[i] = outputs[i].values();
		}

		indexing::sorter idx{grid};
		auto k = [=] __device__ (int tid, int s)
		{
			point z;
			sweep sweep{s, per_sweep, phi, idx};
			auto j = jdata[tid];
			double f = fdata[j];
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + j];
			auto w = sweep.values(z);
			for (auto [i, w]: w | enumerate)
				vdata[tid][i] = w * f;
		};

		auto l = [=] __device__ (int tid, int s)
		{
			sweep sweep{s, per_sweep, phi, idx};
			auto k = kdata[tid];
			auto v = wdata[tid];
			auto j = sweep.indices(k);
			for (auto [i, j]: j | enumerate)
				if (j >= 0) odata[i][j] += res * v[i];
		};

		for (int i = 0; i < sweeps; ++i) {
			{util::transform(k, n, i);}
			{thrust::reduce_by_key(exec, idata, idata+n, vdata, kdata, wdata);}
			{util::transform(l, q, i);}
		}

		for (int i = 1; i < per_sweep; ++i)
			outputs[0] += outputs[i];
		return outputs[0];
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(),
	                                      std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr spread(const grid_tag& tag, const domain_type& domain, delta_type) :
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
			auto unique = uniques(n, indices);
			auto* f = &fdata[n * i];
			return reduce(n, unique, grid, x, f, indices, permutation);
		};
		return map(k, grids, sequence{});
	}
};

using ib::interpolate;

} // namespace novel
} // namespace ib
