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
		int i = s + 0.5;
		auto t = i - s;
		auto q = sqrt(1 - 3 * t * t);
		return 1./3. + (s <= 0.5 ? 2. * q : -3. * t - q) / 6.;
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
	using pattern_type = detail::standard_pattern;
	static constexpr auto
	pattern(std::size_t dimensions)
	{
		return pattern_type{dimensions, meshwidths};
	}
};

} // namespace delta
} // namespace ib
