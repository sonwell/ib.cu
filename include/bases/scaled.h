#pragma once
#include "util/math.h"
#include "differentiation.h"

namespace bases {

template <typename base>
struct scaled : base {
private:
	double scale;
public:
	template <int ... ds>
	constexpr decltype(auto)
	operator()(double r, partials<ds...> p = {}) const
	{
		constexpr auto n = sizeof...(ds);
		return util::math::pow(scale, 2*n) *
			base::operator()(scale * r, p);
	}

	constexpr scaled(const base& phi, double scale) :
		base(phi), scale(scale) {}
};

}
