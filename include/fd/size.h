#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>
#include "util/functional.h"

namespace fd {
namespace __1 {

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
	auto reduce = partial(foldl, std::multiplies<void>{});
	return apply(reduce, sizes(grid, args...));
}

} // namespace __1

using __1::sizes;
using __1::size;

} // namespace fd
