#pragma once
#include <utility>
#include <type_traits>
#include "util/functional.h"
#include "util/launch.h"
#include "fd/correction.h"
#include "fd/discretization.h"
#include "fd/dimension.h"
#include "fd/identity.h"
#include "fd/grid.h"
#include "types.h"

namespace ins {
namespace __1 {

// 1D interpolation to the boundary for pseudo-pressure gradient. The boundary
// conditions for the intermediate velocity field, w⃗, are w⃗ = u⃗ + k∇ɸ.
template <typename lower, typename upper, typename dim_type>
decltype(auto)
boundary(const fd::discretization<fd::dimension<lower, upper>>& comp, const dim_type& dim)
{
	const auto rows = comp.points();
	if (comp == dim)
		return matrix{rows, rows};
	auto solid = comp.solid_boundary;
	return comp.boundary(rows, rows, solid, fd::boundary::lower) +
	       comp.boundary(rows, rows, solid, fd::boundary::upper);
}

} // namespace __1

template <typename grid_type, typename dim_type,
          typename = std::enable_if_t<fd::is_grid_v<grid_type>>>
decltype(auto)
boundary(const grid_type& grid, const dim_type& dim)
{
	using namespace util::functional;
	using fd::correction::zeroth_order;
	struct container { matrix bdy, id; };

	auto k = [&] (const auto& comp) -> container
	{
		return {__1::boundary(comp, dim),
		        identity(comp, zeroth_order)};
	};
	auto op = [] (container l, container r) -> container
	{
		auto& [lb, li] = l;
		auto& [rb, ri] = r;
		auto bdy = kron(lb, ri) + kron(li, rb);
		return {std::move(bdy), kron(li, ri)};
	};
	const auto& components = grid.components();
	auto reduce = partial(foldl, op);
	return apply(reduce, reverse(map(k, components))).bdy;
}

} // namespace ins
