#pragma once
#include <limits>
#include <cmath>

namespace util {
namespace math {
namespace impl {

template <typename value_type,
          typename = std::enable_if_t<std::is_integral_v<value_type>>>
constexpr int
msb(value_type v)
{
	int b = 0;
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
	constexpr auto maxexp = limits::max_exponent;
	constexpr auto bias = maxexp - 1;
	constexpr value_type ln2 = 0.693147180559945309417232121458176568075500134360255254120l;
	constexpr long int emask = (1l << (bits - digits)) - 1;
	typedef union { value_type v; long int i; } map;

	long int logn = 0;
	map m;

	m.v = v;
	if (!(m.i & (emask << (digits-1)))) { // subnormal number
		logn = bias-1;
		v *= limits::max();
		m.v = v;
	}
	auto e = (m.i >> (digits-1)) & emask;
	auto p = e - bias;
	auto q = 2 * bias - e;
	auto pred = q > 0;
	auto r = pred ? q : 1;
	auto s = pred ? 1.0 : 1.0 / (1 << -q);
	m.i = r << (digits-1);
	v *= s * m.v;
	// v now in [1, 2)

	int i = 1;
	value_type c = (v-1) / v;
	value_type z = 0;
	auto y = c;
	auto ln2p = (p-logn)*ln2;

	while (c) {
		auto t = z + y / i;
		if (ln2p + t == ln2p + z) break;
		y *= c;
		i += 1;
		z = t;
	}

	return z+ln2p;
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
	return (long) (x - (x < 0 && (-x != (long) -x)));
}

constexpr double
modulo(double x, double s)
{
	return x - s * floor(x / s);
}

template <typename value_type,
          typename = std::enable_if_t<std::is_integral_v<value_type>>>
constexpr double
pow(double b, value_type e)
{
	using limits = std::numeric_limits<value_type>;
	constexpr auto digits = limits::digits;
	constexpr auto mask0 = (1ull << (digits-1));
	constexpr auto mask = mask0 + (mask0 - 1);
	double v = 1.0;
	for (int i = impl::msb(e & mask); i >= 0; --i) {
		v *= v;
		if (e & (value_type(1) << i))
			v *= b;
	}
	return e < 0 ? 1.0 / v : v;
}

}
}
