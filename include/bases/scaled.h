#pragma once
#include "util/math.h"
#include "differentiation.h"

namespace bases {

template <typename base>
struct scaled : base {
private:
	static_assert(is_basic_function_v<base>);
	double rscale;
	double yscale;
public:
	template <int ... ds>
	constexpr decltype(auto)
	operator()(double r, partials<ds...> p = {}) const
	{
		constexpr auto n = sizeof...(ds);
		return yscale * util::math::pow(rscale, 2*n) *
			base::operator()(rscale * r, p);
	}

	constexpr scaled(const base& phi, double rscale, double yscale = 1.0) :
		base(phi), rscale(rscale), yscale(yscale) {}

	constexpr scaled(const scaled<base>& phi, double rscale, double yscale = 1.0) :
		base(phi), rscale(rscale * phi.rscale), yscale(yscale * phi.rscale) {}
};

template <typename base>
scaled(const scaled<base>&, double, double=1.0) -> scaled<base>;

}
