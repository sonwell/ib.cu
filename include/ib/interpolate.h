#pragma once
#include <thrust/execution_policy.h>
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "sweep.h"
#include "types.h"

namespace ib {

template <typename grid_tag, typename domain_type>
struct interpolate {
public:
	static constexpr auto dimensions = domain_type::dimensions;
	static constexpr auto values_per_sweep = 1 << dimensions;
	static constexpr auto values = 1 << (2 * dimensions);
	static constexpr auto sweeps = (values + values_per_sweep - 1) / values_per_sweep;
private:
	static constexpr thrust::device_execution_policy<thrust::system::cuda::tag> exec = {};

	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp) { return fd::grid{tag, domain, comp}; };
		return map(k, fd::components(domain));
	}

	template <typename grid_type>
	static auto
	accumulate(int n, const grid_type& grid, double* vdata, const matrix& x, const vector& u)
	{
		using point_type = typename grid_type::point_type;
		using sweep_type = sweep<grid_type>;
		auto* xdata = x.values();
		auto* udata = u.values();

		auto k = [=] __device__(int tid)
		{
			double v = 0.0;
			point_type z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto u = grid.units(z);
			auto j = grid.index(u);
			auto d = grid.difference(u);

			for (int i = 0; i < sweeps; ++i) {
				sweep_type sweep(i, grid);
				auto w = sweep.values(d, 1.0);
				auto k = sweep.indices(j);
				for (int l = 0; l < values_per_sweep; ++l)
					if (k[l] >= 0) v += w[l] * udata[k[l]];
			}
			vdata[tid] = v;
		};
		util::transform(k, n);
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(), std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr interpolate(const grid_tag& tag, const domain_type& domain) :
		grids(construct(tag, domain)) {}

	template <typename tuple_type>
	auto
	operator()(int n, const matrix& x, const tuple_type& u) const
	{
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;

		matrix v = 0 * x;
		auto* vdata = v.values();
		auto k = [&] (const auto& grid, const vector& u, auto m)
		{
			static constexpr auto i = decltype(m)::value;
			auto* v = &vdata[n * i];
			return accumulate(n, grid, v, x, u);
		};
		map(k, grids, u, sequence{});
		return v;
	}
};

} // namespace ib
