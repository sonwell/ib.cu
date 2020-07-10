#pragma once
#include "types.h"
#include "grid.h"
#include "util/functional.h"
#include "linalg/kron.h"

namespace fd {
namespace detail {

struct combine_by_sum {
	matrix op;
	matrix id;
};

// Kronecker product
inline decltype(auto)
combine(const matrix& a, const matrix& b)
{
	return kron(a, b);
}

// Kronecker sum
inline combine_by_sum
combine(const combine_by_sum& a, const combine_by_sum& b)
{
	auto& [ao, ai] = a;
	auto& [bo, bi] = b;
	return {kron(ao, bi) + kron(ai, bo), kron(ai, bi)};
}

} // namespace detail

// Combine matrices using either Kronecker sum or product to make
// higher-dimensional operators.
template <typename grid_type, typename k_type>
constexpr decltype(auto)
combine(const grid_type& grid, k_type&& k)
{
	using namespace util::functional;
	auto op = [] (auto&& l, auto&& r) { return detail::combine(l, r); };
	const auto& components = grid.components();
	return apply(partial(foldl, op), reverse(map(k, components)));
}

}
