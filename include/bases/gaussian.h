#pragma once
#include "differentiation.h"

namespace bases {

struct gaussian : differentiable {
	static constexpr auto e = 2.71828182845904523536028747135266249775724709369995;
	double gamma;

	template <int d0, int d1>
	constexpr __host__ __device__ auto
	eval(double r, partials<d0, d1>) const
	{
		return 4 * gamma * gamma * pow(e, -gamma * r * r);
	}

	template <int d0>
	constexpr __host__ __device__ auto
	eval(double r, partials<d0>) const
	{
		return -2 * gamma * pow(e, -gamma * r * r);
	}

	__host__ __device__ auto
	eval(double r, partials<>) const
	{
		return pow(e, -gamma * r * r);
	}

	template <int ... ds>
	__host__ __device__ auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}

	constexpr gaussian(double gamma) : gamma(gamma) {}
};

} // namespace bases
