#pragma once

namespace util {
namespace math {

constexpr __host__ __device__ double
min(double x, double y)
{
	return x < y ? x : y;
}

constexpr __host__ __device__ double
max(double x, double y)
{
	return x < y ? y : x;
}

constexpr __host__ __device__ int
sign(double x)
{
	return (x > 0) - (x < 0);
}

constexpr __host__ __device__ auto
abs(double x)
{
	return sign(x) * x;
}

constexpr __host__ __device__ double
floor(double x)
{
	using limits = std::numeric_limits<double>;
	constexpr auto digits = limits::digits;
	constexpr auto base = static_cast<double>(1ll << digits);

	long long q = abs(x) < base ? 0 : x / base - (x < 0);
	auto r = x - q * base;
	auto f = static_cast<long long>(r - (r < 0));
	return static_cast<double>(f) + q * base;
}

constexpr __host__ __device__ double
modulo(double x, double s)
{
	return x - s * floor(x / s);
}

}
}
