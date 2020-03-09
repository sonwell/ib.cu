#pragma once
#include <cmath>
#include "differentiation.h"

namespace bases {

struct gaussian : basic_function {
private:
	double gamma;

	template <int ... ds>
	__host__ __device__ auto
	eval(double r, partials<ds...>) const
	{
		constexpr auto n = sizeof...(ds);
		return pow(-2 * gamma, n) * exp(-gamma * r * r);
	}
public:
	template <int ... ds>
	__host__ __device__ auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}

	constexpr gaussian(double gamma) : gamma(gamma) {}
};

} // namespace bases
