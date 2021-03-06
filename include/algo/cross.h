#pragma once
#include <array>
#include "util/container.h"
#include "util/permutation.h"

namespace algo {
namespace detail {

template <int sign, int m, int ... ms, std::size_t n,
		 typename ... vector_types>
constexpr void
cross(util::permutation<sign, m, ms...>,
		std::array<double, n>& r, const vector_types& ... vectors)
{
	r[m] += (vectors[ms] * ... * sign);
}

template <typename ... permutations, std::size_t n,
		 typename ... vector_types>
constexpr void
cross(util::container<permutations...>,
		std::array<double, n>& r,
		const vector_types& ... vectors)
{
	((void) cross(permutations{}, r, vectors...), ...);
}

}

// Vector cross product
template <typename ... vector_types>
constexpr auto
cross(const vector_types& ... vectors)
{
	constexpr std::size_t n = sizeof...(vectors) + 1;
	using permutations = util::permutations<n>;
	static_assert(((sizeof(vectors) / sizeof(vectors[0]) == n) && ...),
			"n-1 vectors of size n required for cross product");
	std::array<double, n> r = {0};
	detail::cross(permutations{}, r, vectors...);
	return r;
}

}
