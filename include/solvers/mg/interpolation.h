#pragma once
#include <tuple>
#include <utility>
#include "util/launch.h"
#include "util/functional.h"
#include "util/debug.h"
#include "fd/grid.h"
#include "fd/domain.h"
#include "fd/correction.h"
#include "fd/discretization.h"
#include "solvers/types.h"

namespace solvers {
namespace mg {

// Full weighting restriction and prolongation (interpolation). This is
// hard-coded for 4-point interpolation per dimension.

namespace __1 {

struct interpolation_tag {};
struct restriction_tag {};

struct weights {
public:
	static constexpr auto size() { return 4; }
	constexpr auto operator[](int n) const { return values[n]; }

	constexpr weights(double s) :
		values{generator(-1-s), generator(-s),
			generator(1-s), generator(2-s)} {}
private:
	static constexpr double
	generator(double dx)
	{
		using util::math::abs;
		return 1 - abs(dx)/2;
	}

	double values[4];
};

template <typename lower_type, typename upper_type>
decltype(auto)
interpolation(const fd::discretization<fd::dimension<lower_type, upper_type>>& component,
		interpolation_tag)
{
	static constexpr auto nw = weights::size();
	fd::discretization refinement{component, component.resolution() / 2};
	weights weights(component.shift());
	auto rows = component.points();
	auto cols = refinement.points();
	auto correction = (rows - 2 * cols);
	auto nonzero = nw * cols - 2 + correction;

	sparse::matrix result{rows, cols, nonzero};
	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();
	auto k = [=] __device__ (int tid)
	{
		// This is hard coded for 4 weights
		using util::math::min;
		constexpr int swizzle[] = {3, 1, 2, 0};
		constexpr int shifts[] = {-1, 0, 0, 1};
		if (tid < rows) starts[tid] = tid ? min(1 + 2 * (tid - 1), nonzero-1) : 0;
		auto col = (tid + 1) / nw;
		auto shift = (tid + 1) % nw;
		auto swiz = col + shifts[shift] < cols ? swizzle[shift] : shift;
		auto value = weights[swiz];
		auto index = min(cols-1, col + shifts[shift]) + indexing_base;
		indices[tid] = index;
		values[tid] = value;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);
	using fd::boundary::lower;
	using fd::boundary::upper;
	result += component.interior(rows, cols, weights[correction], upper)
	        + component.interior(rows, cols, weights[3], lower);
	return result;
}

template <typename lower_type, typename upper_type>
decltype(auto)
interpolation(const fd::discretization<fd::dimension<lower_type, upper_type>>& component,
		restriction_tag)
{
	static constexpr auto nw = weights::size();
	fd::discretization refinement{component, component.resolution() / 2};
	weights weights(component.shift());
	auto cols = component.points();
	auto rows = refinement.points();
	auto correction = (cols - 2 * rows);
	auto nonzero = nw * rows - 2 + correction;

	sparse::matrix result{rows, cols, nonzero};
	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();
	auto k = [=] __device__ (int tid)
	{
		if (tid < rows) starts[tid] = tid ? nw * tid - 1 : 0;
		auto shift = (tid + 1) % nw;
		auto value = weights[shift];
		auto row = (tid + 1) / nw;
		auto col = 2 * row + shift - 1;
		indices[tid] = col + indexing_base;
		values[tid] = value / 2;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	using fd::boundary::lower;
	using fd::boundary::upper;
	result += component.interior(rows, cols, weights[0] / 2, lower);
	if (!correction)
		result += component.interior(rows, cols, weights[3] / 2, upper);
	return result;
}

} // namespace __1

template <typename grid_type>
decltype(auto)
interpolation(const grid_type& grid)
{
	using namespace util::functional;
	using matrix = sparse::matrix;
	auto k = [] (const auto& comp)
	{
		constexpr __1::interpolation_tag tag;
		return interpolation(comp, tag);
	};
	auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	return apply(partial(foldl, op), map(k, grid.components()));
}

template <typename grid_type>
decltype(auto)
restriction(const grid_type& grid)
{
	using namespace util::functional;
	using matrix = sparse::matrix;
	auto k = [] (const auto& comp)
	{
		constexpr __1::restriction_tag tag;
		return interpolation(comp, tag);
	};
	auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	return apply(partial(foldl, op), map(k, grid.components()));
}

} // namespace mg
} // namespace solvers
