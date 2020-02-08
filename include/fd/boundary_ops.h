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

namespace fd {
namespace __1 {

template <typename lower_type, typename upper_type, bool is_lower>
decltype(auto)
boundary(const discretization<dimension<lower_type, upper_type>>& component,
		boundary::tag<is_lower> tag)
{
	auto rows = component.points();
	return component.boundary(rows, component.solid_boundary, 1.0, tag);
}

template <typename grid_type, typename view_type, std::size_t n = 0>
decltype(auto)
lower_boundary(const grid_type& grid, const view_type& view,
		correction::order<n> correction = {})
{
	using util::math::pow;
	using namespace util::functional;
	using boundary::lower;
	auto k = [&] (const auto& comp)
	{
		auto r = pow(comp.resolution(), n);
		return comp == view ?
			boundary(comp, lower) * r :
			identity(comp, correction);
	};
	auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	const auto& components = grid.components();
	return apply(partial(foldl, op), reverse(map(k, components)));
}

template <typename grid_type, typename view_type, std::size_t n = 0>
decltype(auto)
upper_boundary(const grid_type& grid, const view_type& view,
		correction::order<n> = {})
{
	using util::math::pow;
	using namespace util::functional;
	using boundary::upper;
	auto k = [&] (const auto& comp)
	{
		auto r = pow(comp.resolution(), n);
		return comp == view ?
			boundary(comp, upper) * r :
			identity(comp, correction::zeroth_order);
	};
	auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	const auto& components = grid.components();
	return apply(partial(foldl, op), reverse(map(k, components)));
}

} // namespace __1

using __1::lower_boundary;
using __1::upper_boundary;

} // namespace fd
