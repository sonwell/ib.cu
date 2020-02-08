#pragma once
#include <cmath>
#include <array>
#include "delta.h"

namespace ib {
namespace delta {

struct hat {
	__host__ __device__ auto
	operator()(double r) const
	{
		auto s = abs(r);
		return 1 - s;
	}
};

template <>
struct traits<hat> {
	static constexpr std::array offsets = {0};
	static constexpr auto calls =
		sizeof(offsets) / sizeof(offsets[0]);
	using rule = detail::rule<1, calls>;
	static constexpr std::array rules = {
		rule{{0.0}, { 1.0}},
		rule{{1.0}, {-1.0}}
	};
	static constexpr auto meshwidths =
		sizeof(rules) / sizeof(rule);
	template <std::size_t dimensions>
		using pattern = standard_pattern<dimensions, meshwidths>;
};

} // namespace delta
} // namespace ib
