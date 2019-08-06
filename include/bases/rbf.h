#pragma once
#include "differentiation.h"

namespace bases {

template <typename basic_function, typename metric>
struct rbf : differentiable {
	basic_function phi;
	metric distance;

	template <std::size_t n>
	constexpr double
	eval(const double (&xs)[n], const double (&xd)[n], partials<>) const
	{
		auto r = distance(xs, xd);
		return phi(r);
	}

	template <std::size_t n, int d>
	constexpr double
	eval(const double (&xs)[n], const double (&xd)[n], partials<d> p) const
	{
		auto r = distance(xs, xd);
		auto rdr = diff(distance, p)(xs, xd);
		return diff(phi, p)(r) * rdr;
	}

	template <std::size_t n, int d0, int d1>
	constexpr double
	eval(const double (&xs)[n], const double (&xd)[n], partials<d0, d1>) const
	{
		partials<d0> p0;
		partials<d1> p1;
		auto r = distance(xs, xd);
		auto rd0r = diff(distance, p0)(xs, xd);
		auto rd1r = d0 == d1 ? rd0r : diff(distance, p1)(xs, xd);
		auto drdr = diff(distance, p0 * p1)(xs, xd);
		return diff(phi, p0 * p1)(r) * rd0r * rd1r + diff(phi, p0)(r) * drdr;
	}

	template <std::size_t n, int ... ds>
	constexpr double
	operator()(const double (&xs)[n], const double (&xd)[n],
			partials<ds...> p = {}) const
	{
		return eval(xs, xd, p);
	}

	constexpr rbf(basic_function phi, metric distance) :
		phi(phi), distance(distance) {}
};

} // namespace bases
