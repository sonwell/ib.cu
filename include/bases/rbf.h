#pragma once
#include "differentiation.h"

namespace bases {

template <typename basic_function, typename metric>
struct rbf : differentiable {
	static_assert(is_basic_function_v<basic_function>,
			"basic functions should inherit from bases::basic_function");
	static_assert(is_metric_v<metric>,
			"metrics should inherit from bases::metric");
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
		auto rdr = distance(xs, xd, p);
		return phi(r, p) * rdr;
	}

	template <std::size_t n, int d0, int d1>
	constexpr double
	eval(const double (&xs)[n], const double (&xd)[n], partials<d0, d1> p) const
	{
		constexpr partials<d0> p0;
		constexpr partials<d1> p1;
		auto r = distance(xs, xd);
		auto rd0r = distance(xs, xd, p0);
		auto rd1r = d0 == d1 ? rd0r : distance(xs, xd, p1);
		auto drdr = distance(xs, xd, p);
		return phi(r, p) * rd0r * rd1r + phi(r, p0) * drdr;
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

template <typename> struct is_rbf : std::false_type {};
template <typename basic_function, typename metric>
struct is_rbf<rbf<basic_function, metric>> : std::true_type {};

template <typename T>
inline constexpr bool is_rbf_v = is_rbf<T>::value;

} // namespace bases
