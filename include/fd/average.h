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

namespace fd {
namespace __1 {

template <typename lower_type, typename upper_type>
auto
average(const discretization<dimension<lower_type, upper_type>>& component)
{
	using util::math::modulo;
	auto src_shift = component.shift();
	alignment shifted(src_shift + 0.5);
	discretization destination{component, shifted};
	auto dst_shift = destination.shift();
	auto skip_one = src_shift > dst_shift;
	auto skip_end = modulo(1-src_shift, 1) < modulo(1-dst_shift, 1);

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
		values[tid] = 0.5;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	if (skip_one) result += component.interior(rows, cols, 0.5, boundary::lower);
	if (skip_end) result += component.interior(rows, cols, 0.5, boundary::upper);
	return result;

}

} // namespace __1

template <typename grid_type, typename view_type,
	typename = std::enable_if_t<is_grid_v<grid_type>>>
decltype(auto)
average(const grid_type& grid, const view_type& view)
{
	using namespace util::functional;
	using correction::zeroth_order;
	auto k = [&] (const auto& comp)
	{
		return comp == view ?
			average(comp) :
			identity(comp, zeroth_order);
	};
	auto op = [] (const matrix& l, const matrix& r) { return kron(l, r); };

	const auto& components = grid.components();
	return apply(partial(foldl, op), reverse(map(k, components)));
}

} // namespace fd
