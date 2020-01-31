#pragma once
#include <cmath>
#include "types.h"
#include "delta.h"

namespace ib {
namespace delta {

struct cosine {
	__host__ __device__ auto
	operator()(double r) const
	{
		constexpr auto pi_halves = M_PI_2;
		return 0.25 * (1 + cos(pi_halves * r));
	}
};

template <>
struct traits<cosine> {
	static constexpr std::array offsets = {-1, 0};
	static constexpr auto calls =
		sizeof(offsets) / sizeof(offsets[0]);
	using rule = detail::rule<1, calls>;
	static constexpr std::array rules = {
		rule{{0.0}, { 1.0,  0.0}},
		rule{{0.0}, { 0.0,  1.0}},
		rule{{0.5}, {-1.0,  0.0}},
		rule{{0.5}, { 0.0, -1.0}}
	};
	static constexpr auto meshwidths =
		sizeof(rules) / sizeof(rules[0]);
	template <std::size_t dimensions>
		using pattern = every_other_pattern<dimensions, meshwidths>;
};

} // namespace delta
} // namespace ib
