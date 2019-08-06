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

template <typename lower, typename upper>
decltype(auto)
boundary(const fd::discretization<fd::dimension<lower, upper>>& component)
{
	const auto rows = component.points();
	auto result = component.boundary(rows, rows, 1.0, fd::boundary::lower)
	            + component.boundary(rows, rows, 1.0, fd::boundary::upper);
	return result;
}

} // namespace __1

template <typename grid_type,
	typename = std::enable_if_t<fd::is_grid_v<grid_type>>>
decltype(auto)
boundary(const grid_type& grid)
{
	using namespace util::functional;
	using fd::correction::zeroth_order;
	struct container {
		matrix boundary;
		matrix identity;
	};

	auto k = [&] (const auto& comp) -> container
	{
		return {__1::boundary(comp), identity(comp, zeroth_order)};
	};
	auto op = [] (const container& l, const container& r) -> container
	{
		auto& [lb, li] = l;
		auto& [rb, ri] = r;
		auto bdy = kron(lb, ri) + kron(li, rb);
		return {std::move(bdy), kron(li, ri)};
	};
	const auto& components = grid.components();
	auto reduce = partial(foldl, op);
	return apply(reduce, reverse(map(k, components))).boundary;

}

} // namespace ins
