#pragma once
#include "util/math.h"
#include "differentiation.h"

namespace bases {

struct unnamed : basic_function {
	template <int d0, int d1>
	constexpr auto
	eval(double r, partials<d0, d1>) const
	{
		using namespace util::math;
		return r ? pow(r, 1/r-6) * (pow(1-log(r), 2) - 3 * r * (1-log(r)) - r*r) : 0;
	}

	template <int d>
	constexpr auto
	eval(double r, partials<d>) const
	{
		using namespace util::math;
		return r ? pow(r, 1/r-3) * (1 - log(r)) : 0;
	}

	constexpr auto
	eval(double r, partials<>) const
	{
		using namespace util::math;
		return r ? pow(r, 1/r) : 0;
	}

	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}
};

} // namespace bases
