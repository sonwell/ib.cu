#pragma once
#include <thrust/execution_policy.h>
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

template <typename domain_type, typename delta_type>
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

	template <typename grid_tag>
	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& ... comp)
			{ return std::array{fd::grid{tag, domain, comp}...}; };
		return apply(k, fd::components(domain));
	}

	template <typename grid_type>
	static auto
	accumulate(int n, const grid_type& grid, double* vdata, const matrix& x, const vector& u)
	{
		using point = ib::point<dimensions>;

		auto* xdata = x.values();
		auto* udata = u.values();

		// Single sweep for all values.
		ib::sweep sweep{0, values, phi, interpolation::sorter{grid}};
		auto k = [=] __device__(int tid)
		{
			point z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];

			auto j = sweep.sort(z);     // sort key
			auto k = sweep.indices(j);  // grid indices
			auto w = sweep.values(z);   // delta values

			double v = 0.0;
			for (auto [k, w]: util::iterators::zip(k, w))
				if (k >= 0) v += w * udata[k]; // k < 0 => error state
			vdata[tid] = v;
		};
		util::transform(k, n);
	}

	auto
	points(const matrix& x) const
	{
		return x.rows() * x.cols() / dimensions;
	}

	using grid_type = fd::grid<domain_type>;
	using grids_type = std::array<grid_type, dimensions>;
	grids_type grids;
public:
	template <typename grid_tag>
	constexpr interpolate(const grid_tag& tag, const domain_type& domain, delta_type) :
		grids(construct(tag, domain)) {}

	template <typename tuple_type>
	void
	operator()(int n, const matrix& x, const tuple_type& ue, matrix& ul) const
	{
		if (n <= 0) return;
		/*
		 * n: number of points in x
		 * x: matrix of positions in [ x_1 x_2 ... x_m y_1 y_2 ... y_m ... ]
		 *    format, where each of x_i, y_i, ... are potentially vectors.
		 * ue: a tuple-like object of (u, v[, w]) fluid velocities at Eulerian
		 *     grid points (hence the e)
		 * ul: the output matrix for Lagrangian velocities (hence the l)
		 */
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;

		auto* vdata = ul.values();
		auto k = [&] (const auto& grid, const vector& u, auto m)
		{
			static constexpr auto i = decltype(m)::value;
			accumulate(n, grid, &vdata[n * i], x, u);
		};
		map(k, grids, ue, sequence{});
	}

	template <typename tuple_type>
	void
	operator()(const matrix& x, const tuple_type& ue, matrix& ul) const
	{
		return operator()(points(x), x, ue, ul);
	}

	template <typename tuple_type>
	matrix
	operator()(int n, const matrix& x, const tuple_type& ue) const
	{
		matrix ul{linalg::size(x)}; // Lagrangian output matrix (hence the l)
		operator()(n, x, ue, ul);
		return ul;
	}

	template <typename tuple_type>
	matrix
	operator()(const matrix& x, const tuple_type& ue) const
	{
		return operator()(points(x), x, ue);
	}
};

} // namespace ib
