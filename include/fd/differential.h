#pragma once
#include <type_traits>
#include <utility>

#include "util/launch.h"
#include "util/functional.h"

#include "types.h"
#include "domain.h"
#include "dimension.h"
#include "discretization.h"
#include "identity.h"
#include "grid.h"
#include "cell.h"
#include "combine.h"

namespace fd {
namespace __1 {

// 1D differential
//
// There are four cases:
//   * periodic: (points just shift around)
//      x---x---x---x--- -> --x---x---x---x-
//            [-1. 1.0 0.0 0.0]
//      A = n [0.0 -1. 1.0 0.0]
//            [0.0 0.0 -1. 1.0]
//            [1.0 0.0 0.0 -1.]
//      --x---x---x---x- -> x---x---x---x---
//            [1.0 0.0 0.0 -1.]
//      A = n [-1. 1.0 0.0 0.0]
//            [0.0 -1. 1.0 0.0]
//            [0.0 0.0 -1. 1.0]
//   * non-periodic
//      ---x---x---x--- -> -x---x---x---x- (number of points increases by 1)
//            [1.0 0.0 0.0]
//      A = n [-1. 1.0 0.0]
//            [0.0 -1. 1.0]
//            [0.0 0.0 -1.]
//      -x---x---x---x- -> ---x---x---x--- (number of points decreases by 1)
//            [-1. 1.0 0.0 0.0]
//      A = n [0.0 -1. 1.0 0.0]
//            [0.0 0.0 -1. 1.0]
template <typename lower_type, typename upper_type>
decltype(auto)
differential(const discretization<fd::dimension<lower_type, upper_type>>& component)
{
	using util::math::modulo;
	auto src_shift = component.shift();
	alignment shifted(src_shift + 0.5);
	discretization destination(component, shifted);
	auto dst_shift = destination.shift();
	auto skip_one = src_shift > dst_shift;
	auto skip_end = modulo(1-src_shift, 1) < modulo(1-dst_shift, 1);
	auto scale = component.resolution();

	auto cols = component.points();
	auto rows = destination.points();
	auto nonzero = rows + cols - 1;
	matrix result{rows, cols, nonzero};

	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();

	auto k = [=] __device__ (int tid)
	{
		if (tid < rows) starts[tid] = tid ? 2 * tid - skip_one : 0;
		auto loc = (tid + skip_one) % 2;
		auto row = (tid + skip_one) / 2;
		auto col = loc + row - skip_one;
		indices[tid] = col + indexing_base;
		values[tid] = scale * (-1 + 2 * loc);
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	if (skip_one) result += component.interior(rows, cols, -scale, boundary::lower);
	if (skip_end) result += component.interior(rows, cols, scale, boundary::upper);
	return result;

}

} // namespace __1

template <typename grid_type, typename view_type,
	typename = std::enable_if_t<is_grid_v<grid_type>>>
decltype(auto)
differential(const grid_type& grid, const view_type& view)
{
	//using namespace util::functional;
	auto k = [&] (const auto& comp)
	{
		using correction::first_order;
		return comp == view ?
			differential(comp) :
			identity(comp, first_order);
	};
	return combine(grid, k);
	//auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	//const auto& components = grid.components();
	//return apply(partial(foldl, op), reverse(map(k, components)));
}

} // namespace fd
