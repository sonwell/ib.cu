#include <iostream>
#include <limits>
#include <type_traits>

namespace impl {

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

	auto q = abs(x) < base ? 0 : floor(x / base);
	auto r = x - q * base;
	auto f = static_cast<long long>(r - (r < 0));
	return static_cast<double>(f) + base * q;
}

constexpr __host__ __device__ double
modulo(double x, double s)
{
	return x - s * floor(x / s);
}

}

int
main(void)
{
	std::cout << std::fixed << impl::floor(-110000000000000.1) << ' ' << impl::floor(-3.2) << std::endl;
	return 0;
}
