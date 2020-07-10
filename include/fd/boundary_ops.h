#pragma once
#include <type_traits>
#include <utility>

#include "util/launch.h"
#include "util/functional.h"
#include "util/math.h"

#include "types.h"
#include "domain.h"
#include "dimension.h"
#include "discretization.h"
#include "identity.h"
#include "boundary.h"
#include "combine.h"

namespace fd {
namespace __1 {

// 1D boundary (either a value at the top or bottom of a 1xn matrix)
template <typename lower_type, typename upper_type, bool is_lower>
decltype(auto)
boundary(const discretization<fd::dimension<lower_type, upper_type>>& component,
		boundary::tag<is_lower> tag)
{
	auto rows = component.points();
	return component.boundary(rows, component.solid_boundary, 1.0, tag);
}

} // namespace __1

template <typename grid_type, typename view_type, std::size_t n = 0>
decltype(auto)
lower_boundary(const grid_type& grid, const view_type& view,
		correction::order<n> correction = {})
{
	auto k = [&] (const auto& comp)
	{
		using util::math::pow;
		using boundary::lower;
		auto r = pow(comp.resolution(), n);
		return comp == view ?
			r * __1::boundary(comp, lower) :
			identity(comp, correction);
	};
	return combine(grid, k);
	//auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	//const auto& components = grid.components();
	//return apply(partial(foldl, op), reverse(map(k, components)));
}

template <typename grid_type, typename view_type, std::size_t n = 0>
decltype(auto)
upper_boundary(const grid_type& grid, const view_type& view,
		correction::order<n> correction = {})
{
	//using namespace util::functional;
	auto k = [&] (const auto& comp)
	{
		using util::math::pow;
		using boundary::upper;
		auto r = pow(comp.resolution(), n);
		return comp == view ?
			r * __1::boundary(comp, upper) :
			identity(comp, correction);
	};
	return combine(grid, k);
	//auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	//const auto& components = grid.components();
	//return apply(partial(foldl, op), reverse(map(k, components)));
}

} // namespace fd
