#pragma once
#include <array>
#include "util/math.h"
#include "polynomials.h"
#include "differentiation.h"

namespace bases {

template <int degree>
struct sinusoids : differentiable {
	template <int ... ds>
	constexpr auto
	operator()(const double (&xs)[1], partials<ds...> p = {}) const
	{
		constexpr typename partials<ds...>::template counts<1> counts;
		constexpr auto d = util::get<0>(counts);
		std::array<double, 1 + 2 * degree> values;
		values[0] = d ? 0.0 : 1.0;
		auto t = xs[0];
		for (int i = 1; i <= degree; ++i) {
			auto p = pow(i, d);
			auto s = sin(i * t);
			auto c = cos(i * t);
			double cyc[] = {s, c, -s, -c};
			values[2 * i - 1 + 0] = cyc[(0 + d) % 4] * p;
			values[2 * i - 1 + 1] = cyc[(1 + d) % 4] * p;
		}
		return values;
	}
};

template <int degree>
struct is_polynomial_basis<sinusoids<degree>> : std::true_type {};

}
