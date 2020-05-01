#pragma once
#include <thrust/execution_policy.h>
#include "cuda/timer.h"
#include "util/log.h"
#include "util/functional.h"
#include "util/iterators.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "delta.h"
#include "types.h"
#include "sweep.h"
#include "indexing.h"

namespace ib {
namespace interpolation {

template <typename grid_type>
struct sorter : indexing::sorter<grid_type> {
	using grid_index_type = indexing::clamped;

	constexpr sorter(grid_type grid) :
		indexing::sorter<grid_type>{std::move(grid)} {}
};

} // namespace interpolation

template <typename grid_tag, typename domain_type, typename delta_type>
struct interpolate {
public:
	static constexpr auto dimensions = domain_type::dimensions;
private:
	using cuda_tag = thrust::system::cuda::tag;
	static constexpr thrust::device_execution_policy<cuda_tag> exec = {};
	static constexpr delta_type phi;
	using traits = delta::traits<delta_type>;
	static constexpr auto values = detail::cpow(traits::meshwidths, dimensions);
	static constexpr auto per_sweep = values;
	static constexpr auto sweeps = (values + per_sweep - 1) / per_sweep;

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
		using point = ib::point<dimensions>;

		auto* xdata = x.values();
		auto* udata = u.values();

		ib::sweep sweep{0, values, phi, interpolation::sorter{grid}};
		auto k = [=] __device__(int tid)
		{
			point z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];

			auto j = sweep.sort(z);
			auto k = sweep.indices(j);
			auto w = sweep.values(z);

			double v = 0.0;
			for (auto [k, w]: util::iterators::zip(k, w))
				if (k >= 0) v += w * udata[k];
			vdata[tid] = v;
		};
		util::transform(k, n);
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(),
	                                      std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr interpolate(const grid_tag& tag, const domain_type& domain, delta_type) :
		grids(construct(tag, domain)) {}

	template <typename tuple_type>
	auto
	operator()(int n, const matrix& x, const tuple_type& u) const
	{
		cuda::timer timer{"ib interpolate"};
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;

		matrix v{linalg::size(x)};
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
