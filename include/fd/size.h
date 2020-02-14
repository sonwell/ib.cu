#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>
#include "util/functional.h"

namespace fd {

template <typename grid_type>
constexpr auto
sizes(const grid_type& grid)
{
	using namespace util::functional;
	auto k = [] (const auto& comp) { return comp.points(); };
	return map(k, grid.components());
}

template <typename grid_type, typename view_type>
constexpr auto
sizes(const grid_type& grid, const view_type& view)
{
	using namespace util::functional;
	auto k = [&] (const auto& comp)
	{
		return comp == view ?
			comp.solid_boundary :
			comp.points();
	};
	return map(k, grid.components());
}

template <typename grid_type, typename ... arg_types>
constexpr auto
size(const grid_type& grid, const arg_types& ... args)
{
	static_assert(sizeof...(arg_types) <= 1);
	using namespace util::functional;
	auto r = [] (auto ... v) { return (v * ... * 1); };
	return apply(r, sizes(grid, args...));
}

} // namespace fd
