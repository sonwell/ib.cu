#pragma once
#include <limits>
#include <cmath>

namespace util {
namespace math {
namespace impl {

constexpr double
log(double x)
{
	int i = 1;
	double c = 1 - x;
	double y = c;
	double z = 0;

	while (true) {
		auto t = z + y / i;
		if (t == z) return t;
		y *= c;
		i += 1;
		z = t;
	}
}

constexpr double
exp(double x)
{
	int i = 1;
	double z = 1;
	double y = x;

	while (true) {
		auto t = z + y;
		if (t == z) return t;
		i += 1;
		y *= x / i;
		z = t;
	}
}

constexpr double
sinusoid(double a, double a0, int i)
{
	constexpr double precomputed[] = {
		1.0,       1.0,
		1.0 / 2,   1.0 / 6,
		1.0 / 12,  1.0 / 20,
		1.0 / 30,  1.0 / 42,
		1.0 / 56,  1.0 / 72,
		1.0 / 90,  1.0 / 110,
		1.0 / 132, 1.0 / 156,
		1.0 / 182, 1.0 / 210
	};
	constexpr auto size = sizeof(precomputed) / sizeof(double);

	double z = a0;
	double y = z;
	double a2 = a * a;

	for (; i < size - 2; i+=2) {
		y *= -precomputed[i+2] * a2;
		auto t = z + y;
		if (t == z) return t;
		z = t;
	}

	do {
		y *= - a2 / ((i+1) * (i+2));
		i += 2;
		auto t = z + y;
		if (t == z) return t;
		z = t;
	} while (true);
}

}

constexpr double
min(double x, double y)
{
	return x < y ? x : y;
}

constexpr double
max(double x, double y)
{
	return x < y ? y : x;
}

constexpr int
sign(double x)
{
	return (x > 0) - (x < 0);
}

constexpr auto
abs(double x)
{
	return sign(x) * x;
}

constexpr double
floor(double x)
{
	return (long long) (x - (x < 0));
}

constexpr double
modulo(double x, double s)
{
	return x - s * floor(x / s);
}

constexpr double
log(double x)
{
	using limits = std::numeric_limits<double>;
	if (x < 0) return limits::quiet_NaN();
	if (x == 0) return -limits::infinity();
	if (x == 1) return 0;
	if (x > 1) return -log(1/x);
	return -impl::log(x);
}

constexpr double
exp(double x)
{
	return impl::exp(x);
}

constexpr double
sqrt(double x)
{
	return exp(0.5 * log(x));
}

constexpr double
pow(double b, double e)
{
	return exp(e * log(b));
}

constexpr double sin(double);

constexpr double
cos(double a)
{
	constexpr auto pi2 = M_PI_2;
	constexpr auto pi4 = M_PI_4;
	auto q = (int) floor((a + pi4) / pi2);
	auto b = a - q * pi2;
	switch ((q % 4 + 4) % 4) {
		case 1: return -impl::sinusoid(b, b, 1);
		case 2: return -impl::sinusoid(b, 1.0, 0);
		case 3: return  impl::sinusoid(b, b, 1);
		default: return impl::sinusoid(b, 1.0, 0);
	}
}

constexpr double
sin(double a)
{
	constexpr auto pi2 = M_PI_2;
	constexpr auto pi4 = M_PI_4;
	auto q = (int) floor((a + pi4) / pi2);
	auto b = a - q * pi2;
	switch (q % 4) {
		case 1: return  impl::sinusoid(b, 1.0, 0);
		case 2: return -impl::sinusoid(b, b, 1);
		case 3: return -impl::sinusoid(b, 1.0, 0);
		default: return impl::sinusoid(b, b, 1);
	}
}

}
}
