#pragma once
#include <cmath>
#include <array>
#include "differentiation.h"

namespace bases {

template <int n>
struct generalized_multiquadric : basic_function {
private:
	double c;

	static constexpr
	std::array<int, 2>
	coefficients(int r)
	{
		int m = n;
		std::array<int, 2> v = {(n+1)%2, n%2};
		for (int i = r; i > 0; --i, m-=2)
			v = {m * v[0], m * v[1] + v[0]};
		return v;
	}

	template <int ... ds>
	__host__ __device__ __forceinline__ auto
	eval(double r, partials<ds...>) const
	{
		constexpr auto eps = std::numeric_limits<double>::min();
		constexpr auto m = sizeof...(ds);
		constexpr auto coeff = coefficients(m);
		auto s2 = r*r + c*c;
		return (0.5 * coeff[0] * log(s2 + eps) + coeff[1]) * pow(s2, n / 2.0 - m);
	}
public:
	template <int ... ds>
	__host__ __device__ __forceinline__ auto
	operator()(double r, partials<ds...> p = {}) const
	{
		return eval(r, p);
	}

	constexpr generalized_multiquadric(double c) : c(c) {}
};

template <int n>
using gmq = generalized_multiquadric<n>;

}
