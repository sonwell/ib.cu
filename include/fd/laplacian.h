#pragma once
#include <type_traits>
#include <utility>

#include "util/launch.h"
#include "util/functional.h"

#include "types.h"
#include "domain.h"
#include "identity.h"
#include "boundary.h"
#include "grid.h"
#include "combine.h"

namespace fd {
namespace __1 {

// 1D 3-point discrete Laplacian operator.
template <typename lower, typename upper>
decltype(auto)
laplacian(const discretization<fd::dimension<lower, upper>>& component)
{
	auto rows = component.points();
	auto n = component.resolution();
	if (!rows) return matrix{rows, rows};

	auto nonzero = 3 * rows - 2;
	matrix result{rows, rows, nonzero};
	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();
	auto scale = n * n;

	auto k = [=] __device__ (int tid)
	{
		if (tid < rows) starts[tid] = tid ? 3 * tid - 1 : 0;
		auto loc = (tid + 1) % 3;
		auto row = (tid + 1) / 3;
		auto col = loc + row - 1;
		indices[tid] = col + indexing_base;
		values[tid] = (1 - 3 * (loc & 1)) * scale;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	result += component.interior(rows, rows, scale, boundary::lower);
	result += component.interior(rows, rows, scale, boundary::upper);
	return result;
}

} // namespace __1

template <typename grid_type,
	typename = std::enable_if_t<is_grid_v<grid_type>>>
decltype(auto)
laplacian(const grid_type& grid)
{
	//using namespace util::functional;
	auto k = [] (const auto& comp) -> detail::combine_by_sum
	{
		using correction::second_order;
		return {laplacian(comp), identity(comp, second_order)};
	};
	return combine(grid, k).op;
	//const auto& components = grid.components();
	//auto reduce = partial(foldl, op);
	//return apply(reduce, reverse(map(k, components))).laplacian;
}

} // namespace fd
