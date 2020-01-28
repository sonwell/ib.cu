#pragma once
#include <cmath>
#include "types.h"
#include "delta.h"

namespace ib {
namespace delta {
namespace detail {

template <std::size_t dimensions, std::size_t meshwidths>
struct every_other_pattern {
	static constexpr auto
	cpow(int base, int exp)
	{
		if (exp == 0) return 1;
		if (base == 0) return 0;
		if (exp == 1) return base;
		auto r = cpow(base, exp >> 1);
		return cpow(base, exp & 1) * r * r;
	}

	static constexpr auto
	div(int x, int y)
	{
		return std::make_pair(x / y, x % y);
	}

	constexpr auto
	operator()(int offset, int i) const
	{
		constexpr auto shift = (meshwidths - 1) >> 1;
		constexpr auto half_meshwidths = (meshwidths + 1) >> 1;
		constexpr auto mod = cpow(half_meshwidths, dimensions);

		auto [x, y] = div(offset + i, mod);
		ib::shift<dimensions> s{};
		for (int j = 0; j < dimensions; ++j) {
			s[j] = x % 2 + 2 * (y % half_meshwidths) - shift;
			x /= 2;
			y /= half_meshwidths;
		}
		return s;
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
		using pattern = detail::every_other_pattern<dimensions, meshwidths>;
};

} // namespace delta
} // namespace ib
