#pragma once
#include "util/math.h"

namespace algo {
namespace impl {

template <typename T>
requires std::is_floating_point_v<T>
constexpr T
gcd(T a, T b)
{
	// Euclid's GCD algorithm modified for floating point values
	using limits = std::numeric_limits<T>;
	auto eps = limits::epsilon();

	while (util::math::abs(b) > eps) {
		if (a < b) std::swap(a, b);
		auto c = a - util::math::floor(a / b) * b;
		a = b; b = c;
		eps *= 2;
	}
	return a;
}

} // namespace impl

template <typename T>
requires std::is_floating_point_v<T>
constexpr T
gcd(T a, T b)
{
	if (a < b) std::swap(a, b);
	return impl::gcd(1.0, b / a) * a;
}

template <typename T>
requires std::is_integral_v<T>
constexpr T
gcd(T a, T b)
{
	if (a == 0) return b;
	if (a == 1) return 1;
	if (a > b) std::swap(a, b);
	return gcd(b % a, a);
}

} // namespace algo
