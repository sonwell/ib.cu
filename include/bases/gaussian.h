#pragma once
#include "util/math.h"
#include "differentiation.h"

namespace bases {

struct gaussian : differentiable {
private:
	double gamma;

	template <int ... ds>
	constexpr auto
	eval(double r, partials<ds...>) const
	{
		using util::math::pow;
		using util::math::exp;
		constexpr auto n = sizeof...(ds);
		return pow(-2 * gamma, n) * exp(-gamma * r * r);
	}
public:
	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}

	constexpr gaussian(double gamma) : gamma(gamma) {}
};

} // namespace bases
