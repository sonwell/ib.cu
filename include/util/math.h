#pragma once
#include <limits>
#include <cmath>

namespace util {
namespace math {
namespace impl {

template <typename value_type,
          typename = std::enable_if_t<std::is_integral_v<value_type>>>
constexpr unsigned
msb(value_type v)
{
	unsigned b = 0;
	for (auto bits = sizeof(v) * 4; bits > 0; bits >>= 1)
		if (v >> bits) {
			b += bits;
			v >>= bits;
		}
	return b;
}

template <typename value_type,
          typename = std::enable_if_t<
			  std::numeric_limits<value_type>::is_iec559>>
constexpr auto
log(value_type v)
{
	using limits = std::numeric_limits<value_type>;
	constexpr auto bits = sizeof(value_type) * 8;
	constexpr auto digits = limits::digits;
	constexpr auto minexp = limits::min_exponent;
	constexpr auto maxexp = limits::max_exponent;
	constexpr value_type loge = 1.442695040888963407359924681001892137426645954152985934135l;
	constexpr long int emask = (1l << (bits - digits)) - 1;
	typedef union { value_type v; long int i; } map;

	long int logn = 0;
	map m;

	m.v = v;
	if (!(m.i & (emask << (digits-1)))) { // subnormal number
		logn = maxexp-2;
		v *= limits::max();
		m.v = v;
	}
	auto e = (m.i >> (digits-1)) & emask;
	auto logb = maxexp-1 - e;
	auto diff = logb - minexp;
	long int n[] = {diff+1, 1l};
	double s[] = {1.0, 1.0 / (1 << -diff)};
	m.i = n[diff < 0] << (digits-1);
	v *= s[diff < 0] * m.v;
	// v now in (0.5, 2)

	int i = 1;
	value_type c = 1 - v;
	value_type z = 0;
	auto y = c;

	while (c) {
		auto t = z + y / i;
		if (t == z) break;
		y *= c;
		i += 1;
		z = t;
	}

	return z-(logn+logb+1)/loge;
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
	return (long) (x - (x < 0));
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
	return impl::log(x);
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

constexpr double sin(double);
constexpr double cos(double);

constexpr double
pow(double b, double e)
{
	int p = floor(e);
	if (e != p) return std::numeric_limits<double>::quiet_NaN();
	return (1 - 2 * (p & 1)) * exp(e * log(b));
}

template <typename value_type,
          typename = std::enable_if_t<std::is_integral_v<value_type>>>
constexpr double
pow(double b, value_type e)
{
	double v = 1.0;
	for (auto i = msb(e); i >= 0; --i) {
		v *= v;
		if (e & (value_type(1) << i))
			v *= b;
	}
	return v;
}


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
