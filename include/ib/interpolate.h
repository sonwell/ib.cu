#pragma once
#include <thrust/execution_policy.h>
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "delta.h"
#include "novel.h"
#include "types.h"

namespace ib {
namespace novel {

using fd::__1::delta;
using fd::__1::shift;

template <std::size_t dimensions>
struct interpolate_info {
	static constexpr auto total_values = 1 << (2 * dimensions);
	static constexpr auto values_per_sweep = 1 << dimensions;
	static constexpr auto sweeps = (total_values + values_per_sweep -1) / values_per_sweep;
	using container_type = values_container<values_per_sweep>;
};

template <typename grid_type>
struct interpolate_sweep {
	static constexpr auto dimensions = grid_type::dimensions;
	using info = interpolate_info<dimensions>;
	static constexpr auto values_per_sweep = info::values_per_sweep;
	static constexpr auto total_values = info::total_values;
	static constexpr auto sweeps = info::sweeps;
	using container_type = typename info::container_type;
	static constexpr cosine_delta phi = {};

	int count;
	const grid_type& grid;

	static constexpr auto mask(int i, int m, int n) { return (i & (m << n)) >> n; }

	constexpr auto
	values(const delta<dimensions>& dx, double f) const
	{
		container_type values = {0.0};
		if (count >= sweeps)
			return values;

		double weights[dimensions][2];
		for (int i = 0; i < dimensions; ++i) {
			auto base = mask(count, 1, i) - 1;
			auto v = phi(base + dx[i]);
			weights[i][0] = v;
			weights[i][1] = 0.5 - v;
		}

		for (int i = 0; i < values_per_sweep; ++i) {
			double v = f;
			for (int j = 0; j < dimensions; ++j)
				v *= weights[j][mask(i, 1, j)];
			values[i] = v;
		}

		return values;
	}

	constexpr auto
	indices(int index) const
	{
		std::array<int, values_per_sweep> values = {0};
		indexer idx{grid};

		auto indices = idx.decompose(index);
		for (int i = 0; i < values_per_sweep; ++i) {
			auto base = mask(count, 1, i) - 1;
			shift<dimensions> s = {0};
			for (int j = 0; j < dimensions; ++j)
				s[j] = base + 2 * mask(i, 3, j);
			values[i] = idx.grid(indices + s);
		};

		return values;
	}

	constexpr interpolate_sweep(int count, const grid_type& grid) :
		count(count), grid(grid) {}
};

template <typename grid_tag, typename domain_type>
struct interpolate {
public:
	static constexpr auto dimensions = domain_type::dimensions;
	using info = interpolate_info<dimensions>;
	static constexpr auto values_per_sweep = info::values_per_sweep;
	static constexpr auto total_values = info::total_values;
	static constexpr auto sweeps = info::sweeps;
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
		using sweep_type = interpolate_sweep<grid_type>;
		auto* xdata = x.values();
		auto* udata = u.values();

		auto k = [=] __device__(int tid)
		{
			double v = 0.0;
			point_type z;
			indexer idx{grid};
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto u = grid.units(z);
			auto d = grid.difference(u);
			auto j = idx.sort(u);

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

} // namespace novel

using novel::interpolate;

} // namespace ib
