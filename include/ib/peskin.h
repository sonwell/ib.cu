#pragma once
#include "delta.h"

namespace ib {
namespace delta {
struct peskin {
	__host__ __device__ auto
	operator()(double r) const {
		auto s = abs(r);
		return s < 1 ?
			(3 - 2 * s + sqrt(1 + 4 * s - 4 * s * s)) / 8 :
			(5 - 2 * s - sqrt(-7 + 12 * s - 4 * s * s)) / 8;
	}
};

template <>
struct traits<peskin> {
	static constexpr std::array offsets = {0};
	static constexpr auto calls =
		sizeof(offsets) / sizeof(offsets[0]);
	using rule = detail::rule<2, calls>;
	static constexpr std::array rules = {
		rule{{ 0.75,  0.5}, {-1.0}},
		rule{{ 0.00,  0.0}, { 1.0}},
		rule{{-0.25, -0.5}, { 1.0}},
		rule{{ 0.50,  0.0}, {-1.0}}
	};
	static constexpr auto meshwidths =
		sizeof(rules) / sizeof(rules[0]);
	template <std::size_t dimensions>
		using pattern = every_other_pattern<dimensions, meshwidths>;
};

}
}
