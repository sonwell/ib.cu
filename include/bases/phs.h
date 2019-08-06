#pragma once
#include "differentiation.h"

namespace bases {

template <unsigned int n>
struct polyharmonic_spline : differentiable {
	static constexpr auto eps = std::numeric_limits<double>::epsilon();

	template <int d0, int d1>
	constexpr auto
	eval(double r, partials<d0, d1>) const
	{
		if constexpr (n % 2 == 0)
			return (n + (n-2) * (n * log(r + eps) + 1)) * pow(r, n-4);
		else
			return n * (n-2) * pow(r, n-4);
	}

	template <int d0>
	constexpr auto
	eval(double r, partials<d0>) const
	{
		if constexpr (n % 2 == 0)
			return (n * log(r + eps) + 1) * pow(r, n-2);
		else
			return n * pow(r, n-2);
	}

	constexpr auto
	eval(double r, partials<>) const
	{
		if constexpr (n % 2 == 0)
			return log(r + eps) * pow(r, n);
		else
			return pow(r, n);
	}

	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}
};

} // namespace bases
