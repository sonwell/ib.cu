#pragma once
#include <type_traits>
#include <utility>
#include <tuple>
#include "util/launch.h"
#include "util/functional.h"


#include "types.h"
#include "cell.h"
#include "grid.h"
#include "correction.h"
#include "domain.h"
#include "discretization.h"
#include "combine.h"

namespace fd {
namespace __1 {

using correction::order;

// 1D identity is a sparse diagonal matrix with 1s on the diagonal.
// Modifications are made in the upper left and lower right entries in some
// instances. See discretization.h for details.
template <typename lower, typename upper, std::size_t n>
auto
identity(const discretization<fd::dimension<lower, upper>>& view,
		const order<n>& correction)
{
	const auto rows = view.points();
	matrix result{rows, rows, rows};

	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();
	auto [lw, uw] = view.coefficient(correction);

	auto k = [=, lw=lw, uw=uw] __device__ (int tid)
	{
		double value = 1.0;
		if (tid == 0) value += lw;
		if (tid+1 == rows) value += uw;

		starts[tid] = tid;
		indices[tid] = tid + indexing_base;
		values[tid] = value;
		if (!tid) starts[rows] = rows;
	};
	util::transform<128, 7>(k, rows);
	return result;
}

} // namespace __1

template <typename grid_type, std::size_t n,
	typename = std::enable_if_t<is_grid_v<grid_type>>>
auto
identity(const grid_type& grid, __1::order<n> correction)
{
	//using namespace util::functional;
	auto k = [&] (const auto& view) { return identity(view, correction); };
	return combine(grid, k);
	//auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };
	//const auto& components = grid.components();
	//return apply(partial(foldl, op), reverse(map(k, components)));
}

} // namespace fd
