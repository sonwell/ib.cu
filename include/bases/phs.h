#pragma once
#include <array>
#include <cmath>
#include "differentiation.h"

namespace bases {

template <unsigned int n>
struct polyharmonic_spline : basic_function {
private:
	static constexpr auto eps = std::numeric_limits<double>::min();

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
		constexpr auto m = sizeof...(ds);
		static_assert(n - 2 * m >= 0, "attempting to use a polyharmonic "
				"spline that is not as smooth as needed. Increase the "
				"order of your polyharmonic spline and try again.");
		constexpr auto coeffs = coefficients(m);
		return (coeffs[0] * log(r + eps) + coeffs[1]) * pow(r, n-2*m);
	}
public:
	template <int ... ds>
	__host__ __device__ __forceinline__ auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}
};

} // namespace bases
