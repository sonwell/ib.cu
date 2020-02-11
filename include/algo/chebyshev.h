#pragma once
#include <utility>
#include <cmath>
#include "types.h"
#include "gershgorin.h"
#include "preconditioner.h"

namespace algo {
namespace impl {

template <std::size_t n>
struct polynomial {
	std::array<double, n> coeffs;

	constexpr double
	operator()(double a, double b) const
	{
		double v = 0.0;
		for (int i = 0; i < n; ++i)
			v += coeffs[i] * pow(a, i) * pow(b, n-1-i);
		return v;
	}

	template <typename ... arg_types>
	constexpr polynomial(arg_types ... args) :
		coeffs{(double) args...} {}
};

template <typename ... arg_types>
polynomial(arg_types&& ...) -> polynomial<sizeof...(arg_types)>;

template <int degree> struct chebw;

/*
 * denominator = polynomial(a, b, (2n)!/(2n)!, (2n)!/(2!*(2n-2)!), ...,
 *                          (2n)!/((2k)!*(2n-2k)!), ..., (2n)!/(2n)!).
 */

template <>
struct chebw<1> {
	static constexpr polynomial p{2};
	static constexpr polynomial q{1, 1};
	double weights[1];

	constexpr chebw(double a, double b) :
		weights {p(a, b) / q(a, b)} {}
};

template <>
struct chebw<2> {
	static constexpr polynomial p1{8};
	static constexpr polynomial p2{8, 8};
	static constexpr polynomial q{1, 6, 1};
	double weights[2];

	constexpr chebw(double a, double b) :
		weights {
			p1(a, b) / q(a, b),
			p2(a, b) / q(a, b)
		} {}
};

template <>
struct chebw<3> {
	static constexpr polynomial p1{32};
	static constexpr polynomial p2{48, 48};
	static constexpr polynomial p3{18, 60, 18};
	static constexpr polynomial q{1, 15, 15, 1};
	double weights[3];

	constexpr chebw(double a, double b) :
		weights{
			p1(a, b) / q(a, b),
			p2(a, b) / q(a, b),
			p3(a, b) / q(a, b)
		} {}
};

template <>
struct chebw<4> {
	static constexpr polynomial p1{128};
	static constexpr polynomial p2{256, 256};
	static constexpr polynomial p3{160, 448, 160};
	static constexpr polynomial p4{32, 224, 224, 32};
	static constexpr polynomial q{1, 28, 70, 28, 1};
	double weights[4];

	constexpr chebw(double a, double b) :
		weights{
			p1(a, b) / q(a, b),
			p2(a, b) / q(a, b),
			p3(a, b) / q(a, b),
			p4(a, b) / q(a, b)
		} {}
};

} // namespace impl

class chebyshev {
private:
	impl::chebw<4> chebw;

	auto
	polynomial(vector r) const
	{
		const auto& weights = chebw.weights;
		static constexpr auto num_weights = sizeof(weights) / sizeof(double);

		auto s = weights[0] * r;
		for (int i = 1; i < num_weights; ++i) {
			vector t = r;
			gemv(-1.0, m, s, weights[i], t);
			swap(s, t);
		}
		return s;
	}
protected:
	matrix m;
public:
	auto operator()(vector r) const { return polynomial(std::move(r)); }
	chebyshev(double a, double b, matrix m) : chebw(a, b), m(m) {}
	chebyshev(chebyshev&& o) : chebw(std::move(o.chebw)), m(std::move(o.m)) {}
};

} // namespace algo
