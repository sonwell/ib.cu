#pragma once
#include <cmath>
#include "delta.h"

namespace ib {
namespace delta {
namespace detail {

struct every_other_pattern {
	std::size_t dimensions;
	std::size_t meshwidths;

	static constexpr auto
	cpow(int base, int exp)
	{
		if (exp == 0) return 1;
		if (base == 0) return 0;
		if (exp == 1) return base;
		auto r = cpow(base, exp >> 1);
		return cpow(base, exp & 1) * r * r;
	}

	constexpr auto
	operator[](int n) const
	{
		auto half_meshwidths = meshwidths / 2;
		auto mod = cpow(half_meshwidths, dimensions);
		auto inner = n % mod;
		auto outer = n / mod;
		auto key = 0;
		auto weight = 1;
		for (int i = 0; i < dimensions; ++i) {
			key += weight * (outer % 2 + 2 * (inner % half_meshwidths));
			inner /= half_meshwidths;
			outer /= 2;
			weight *= 2 * half_meshwidths;
		}
		return key;
	}
};

} // namespace detail

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
	using pattern_type = detail::every_other_pattern;
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
	static constexpr auto
	pattern(std::size_t dimensions)
	{
		return pattern_type{dimensions, meshwidths};
	}
};

} // namespace delta
} // namespace ib
