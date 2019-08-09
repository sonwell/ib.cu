#pragma once
#include "util/math.h"

namespace algo {
namespace impl {

constexpr void
swap(double& a, double& b)
{
	double c = a;
	a = b;
	b = c;
}

constexpr double
gcd(double a, double b)
{
	using util::math::abs;
	using limits = std::numeric_limits<double>;
	constexpr auto eps = limits::epsilon();

	while (abs(b) > eps) {
		if (a < b) impl::swap(a, b);
		auto c = a - ((int) (a / b)) * b;
		a = b; b = c;
	}
	return a;
}

} // namespace impl

constexpr double
gcd(double a, double b)
{
	if (a < b) impl::swap(a, b);
	return impl::gcd(1.0, b / a) * a;
}

} // namespace algo