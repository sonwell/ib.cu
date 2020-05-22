#pragma once
#include <cmath>
#include <array>
#include "delta.h"

namespace ib {
namespace delta {

struct roma {
	__host__ __device__ auto
	operator()(double r) const
	{
		auto s = abs(r);
		return s >= 1.5 ? 0 : s < 0.5 ?
			(1 + sqrt(1-3*s*s)) / 3 :
			(5 - 3*s - sqrt(1-3*(1-s)*(1-s))) / 6;
	}
};

template <>
struct traits<roma> {
	static constexpr std::array offsets = {0};
	static constexpr auto calls =
		sizeof(offsets) / sizeof(offsets[0]);
	using rule = detail::rule<2, calls>;
	static constexpr std::array rules = {
		rule{{0.5, -0.5}, {-0.5}},
		rule{{0.0,  0.0}, { 1.0}},
		rule{{0.5,  0.5}, {-0.5}}
	};
	static constexpr auto meshwidths =
		sizeof(rules) / sizeof(rules[0]);
	template <std::size_t dimensions>
		using pattern = standard_pattern<dimensions, meshwidths>;
};

} // namespace delta
} // namespace ib
