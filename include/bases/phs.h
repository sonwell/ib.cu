#pragma once
#include <array>
#include <cmath>
#include "differentiation.h"

namespace bases {

// Polyharmonic spline RBF
//
//     ɸ(r) = r^n, n odd
//     ɸ(r) = r^n log r, n even
//
template <unsigned int n>
struct polyharmonic_spline : basic_function {
private:
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
		static_assert(n >= 2 * m, "attempting to use a polyharmonic "
				"spline that is not as smooth as needed. Increase the "
				"order of your polyharmonic spline and try again.");
		constexpr auto coeffs = coefficients(m);
		return (coeffs[0] * log(r + eps) + coeffs[1]) * pow(r, n - 2*m);
	}
public:
	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = {}) const
	{
		return eval(r, p);
	}
};

template <unsigned int n>
using phs = polyharmonic_spline<n>;

} // namespace bases
