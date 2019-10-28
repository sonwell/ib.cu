#pragma once
#include <thrust/execution_policy.h>
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "fd/grid.h"
//#include "sweep.h"
#include "delta.h"
#include "novel.h"
#include "types.h"

namespace ib {
namespace novel {

template <typename grid_tag, typename domain_type>
struct spread {
public:
	static constexpr auto dimensions = domain_type::dimensions;
	using info = sweep_info<dimensions>;
	static constexpr auto values_per_sweep = info::values_per_sweep;
	static constexpr auto total_values = info::total_values;
	static constexpr auto sweeps = info::sweeps;
	using container_type = typename info::container_type;
private:
	static constexpr thrust::device_execution_policy<thrust::system::cuda::tag> exec = {};

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
	reduce(int n, int q, const grid_type& grid, const matrix& x, const double* fdata,
			const util::memory<int>& indices, const util::memory<int>& permutation)
	{
		using namespace util::functional;
		using point_type = typename grid_type::point_type;
		using sweep_type = sweep<grid_type>;
		constexpr auto gr = [] (const auto& c) { return c.resolution(); };
		constexpr auto prod = partial(foldl, std::multiplies<double>{}, 1.0);
		auto res = apply(prod, map(gr, grid.components()));
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

		vector outputs[sweeps];
		double* odata[sweeps];
		for (int i = 0; i < sweeps; ++i) {
			outputs[i] = vector(size, linalg::zero);
			odata[i] = outputs[i].values();
		}

		auto k = [=] __device__ (int tid, int s)
		{
			sweep_type sweep(s, grid);
			auto j = jdata[tid];
			point_type z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + j];
			double f = fdata[j];
			auto u = grid.units(z);
			auto d = grid.difference(u);
			vdata[tid] = sweep.values(d, f);
		};

		auto l = [=] __device__ (int tid, int s)
		{
			sweep_type sweep(s, grid);
			auto k = kdata[tid];
			auto v = wdata[tid];
			auto j = sweep.indices(k);

			for (int i = 0; i < values_per_sweep; ++i)
				if (j[i] >= 0) odata[i][j[i]] += res * v[i];
		};

		for (int i = 0; i < sweeps; ++i) {
			util::transform(k, n, i);
			thrust::reduce_by_key(exec, idata, idata+n, vdata, kdata, wdata);
			util::transform(l, q, i);
		}

		for (int i = 1; i < sweeps; ++i)
			outputs[0] += outputs[i];
		return outputs[0];
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
			auto unique = uniques(n, indices);
			auto* f = &fdata[n * i];
			return reduce(n, unique, grid, x, f, indices, permutation);
		};
		return map(k, grids, sequence{});
	}
};

} // namespace novel

using novel::spread;

} // namespace ib
