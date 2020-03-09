#pragma once
#include <cmath>
#include "differentiation.h"

namespace bases {

template <int n>
struct gmq : basic_function {
private:
	double c;

	static constexpr int
	coefficient(int r)
	{
		int s = 1;
		for (int i = 0; i < r; ++i)
			s *= n - 2*r;
		return s;
	}

	template <int ... ds>
	__host__ __device__ __forceinline__ auto
	eval(double r, partials<ds...>) const
	{
		constexpr auto m = sizeof...(ds);
		constexpr auto coeff = coefficient(m);
		return coeff * pow(r*r + c*c, n / 2.0 - m) / pow(1 + c*c, n / 2.0);
	}

public:
	template <int ... ds>
	__host__ __device__ __forceinline__ auto
	operator()(double r, partials<ds...> p = {}) const
	{
		return eval(r, p);
	}

	constexpr gmq(double c) : c(c) {}
};

template <int n>
using generalized_multiquadric = gmq<n>;

}
