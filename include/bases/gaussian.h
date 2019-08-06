#pragma once
#include "util/math.h"
#include "differentiation.h"

namespace bases {

struct gaussian : differentiable {
	double gamma;

	template <int d0, int d1>
	constexpr auto
	eval(double r, partials<d0, d1>) const
	{
		using util::math::exp;
		return 4 * gamma * gamma * exp(-gamma * r * r);
	}

	template <int d0>
	constexpr auto
	eval(double r, partials<d0>) const
	{
		using util::math::exp;
		return -2 * gamma * exp(-gamma * r * r);
	}

	constexpr auto
	eval(double r, partials<>) const
	{
		using util::math::exp;
		return exp(-gamma * r * r);
	}

	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		using namespace util::math;
		return eval(r, p);
	}

	constexpr gaussian(double gamma) : gamma(gamma) {}
};

} // namespace bases
