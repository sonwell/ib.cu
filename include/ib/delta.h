#pragma once
#include <array>
#include "types.h"

namespace ib {
namespace delta {

/* The traits object lists information about what order to access the values of
 * the delta function, how to avoid extraneous calls to the delta function, etc.
 *
 * A highly-optimized 4-point delta function with the even-odd splitting can be
 * sped up by a factor of about 3 if it is _hardcoded_. Unfortunately, I have
 * not been able to generalize the code to arbitrary deltas, so instead of
 * hardcoding it for 4-point deltas, only the access pattern attribute `pattern`
 * is used in the IB operations, and there are as many calls to the delta
 * function as there are dimensions in the domain. */
template <typename> struct traits;

namespace detail {

/* A polynomial in r is used to define a relationship between one delta function
 * value and some shifted values. For example, the relationships from Peskin
 * (2002) for -1 < r <= 0,
 *
 *              ϕ(r)          +  ϕ(r+2) = 1/2
 *     ϕ(r-1)        + ϕ(r+1)           = 1/2
 *     ϕ(r-1)        - ϕ(r+1) - 2ϕ(r+2) = r
 *
 * can be rearranged to give
 *
 *     ϕ(r-1) =  (3/4 + r/2) - ϕ(r)
 *     ϕ(r)   =  (  0 +  0r) + ϕ(r)
 *     ϕ(r+1) = -(1/4 + r/2) + ϕ(r)
 *     ϕ(r+2) =  (1/2 +  0r) - ϕ(r)
 *
 * i.e., we need only evaluate ϕ(r) to get all 4 values. The polynomials are
 * those in the parentheses (plus a sign). In this case, we only need to make
 * one call to ϕ to get all values. Other deltas may need additional calls to
 * ϕ. We construct a linear system like the above for the delta function and
 * the coefficients of the calls to ϕ are stored in a vec object. Together, the
 * polynomial and vec comprise a rule object, which holds information for one
 * equation of the linear system, e.g.,
 *
 *     rule<2, 1>{{ 0.75,  0.5}, {-1.0}}
 *                ^~~~~~~~~~~~   ^~~~~
 *                  polynomial     vec
 *
 * for the first equation.
 *
 * NB: This is not currently used, but they have been defined for each delta
 * function (I think). -- Andy, Jul 7 '21
 * */

template <std::size_t n>
struct polynomial {
	std::array<double, n> weights;

	constexpr double
	operator()(double r) const
	{
		double acc = 0.0;
		double p = 1.0;
		for (int i = 0; i < n; ++i, p *= r)
			acc += p * weights[i];
		return acc;
	}
};

template <std::size_t m>
struct vec {
	std::array<double, m> weights;

	constexpr double
	operator*(const vec<m>& c) const
	{
		double acc = 0.0;
		for (int i = 0; i < m; ++i)
			acc += weights[i] * c[i];
		return acc;
	}

	constexpr const double&
	operator[](int i) const
	{
		return weights[i];
	}
};

template <std::size_t n, std::size_t m>
struct rule {
	polynomial<n> p;
	vec<m> r;
};

} // namespace detail

template <std::size_t dimensions, std::size_t meshwidths>
struct standard_pattern {
	constexpr auto
	operator()(int offset, int i) const
	{
		constexpr auto shift = (meshwidths - 1) >> 1;
		auto n = offset + i;
		ib::shift<dimensions> s{};
		for (int j = 0; j < dimensions; ++j) {
			s[j] = n % meshwidths - shift;
			n /= meshwidths;
		}
		return s;
	}
};

template <std::size_t dimensions, std::size_t meshwidths>
struct every_other_pattern {
	static constexpr auto
	cpow(int base, int exp)
	{
		if (exp == 0) return 1;
		if (base == 0) return 0;
		if (exp == 1) return base;
		auto r = cpow(base, exp >> 1);
		return cpow(base, exp & 1) * r * r;
	}

	static constexpr auto
	div(int x, int y)
	{
		return std::make_pair(x / y, x % y);
	}

	constexpr auto
	operator()(int offset, int i) const
	{
		constexpr auto shift = (meshwidths - 1) >> 1;
		constexpr auto half_meshwidths = (meshwidths + 1) >> 1;
		constexpr auto mod = cpow(half_meshwidths, dimensions);

		auto [x, y] = div(offset + i, mod);
		ib::shift<dimensions> s{};
		for (int j = 0; j < dimensions; ++j) {
			s[j] = x % 2 + 2 * (y % half_meshwidths) - shift;
			x /= 2;
			y /= half_meshwidths;
		}
		return s;
	}
};

} // namespace delta
} // namespace ib
