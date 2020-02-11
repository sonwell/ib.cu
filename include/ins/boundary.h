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

template <typename lower, typename upper, typename dim_type>
decltype(auto)
boundary(const fd::discretization<fd::dimension<lower, upper>>& comp, const dim_type& dim)
{
	const auto rows = comp.points();
	if (comp == dim)
		return matrix{rows, rows};
	return fd::__1::single_entry(rows, rows, comp.solid_boundary, {false, false, 1.0})
	     + fd::__1::single_entry(rows, rows, comp.solid_boundary, {true, true, 1.0});
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
