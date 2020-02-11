#pragma once
#include <array>
#include "util/math.h"
#include "differentiation.h"

namespace bases {

template <unsigned int n>
struct polyharmonic_spline : differentiable {
private:
	static constexpr auto eps = std::numeric_limits<double>::min();

	template <int m, std::size_t r>
	struct coefficients {
		static constexpr auto
		values(std::array<int, 2> v)
		{
			if constexpr (r == 0)
				return v;
			else
				return coefficients<m-2, r-1>::values({m*v[0], v[0]+m*v[1]});
		}
	};

	template <int ... ds>
	constexpr auto
	eval(double r, partials<ds...>) const
	{
		using util::math::log;
		using util::math::pow;
		constexpr auto m = sizeof...(ds);
		static_assert(n - 2 * m >= 0, "attempting to use a polyharmonic "
				"spline that is not as smooth as needed. Increase the "
				"order of your polyharmonic spline and try again.");
		using coeff_type = coefficients<n, m>;
		constexpr auto coeffs = coeff_type::values({(n+1)%2, n%2});

		return (coeffs[0] * log(r + eps) + coeffs[1]) * pow(r, n-2*m);
	}
public:
	template <int ... ds>
	constexpr auto
	operator()(double r, partials<ds...> p = partials<>()) const
	{
		return eval(r, p);
	}
};

} // namespace bases
