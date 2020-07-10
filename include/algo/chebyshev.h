#pragma once
#include <utility>
#include <cmath>
#include "types.h"
#include "gershgorin.h"
#include "preconditioner.h"

namespace algo {

// Chebyshev iteration
//
// Suppose x satisfies Ax = b for positive (negative) semidefinite A. Let x_1 be
// a guess for x. We can construct an improved guess, x_2, via
//
//     x_2 = x_1 + p(A)(b - Ax_1).
//
// Subtracting x from both sides, we have
//
//     e_2 = e_1 + p(A) r_1
//         = e_1 + p(A) Ae_1
//         = (I + p(A) A) e_1
//        := q(A) e_1
//
// NB: q(0) = 1
//
// For a reduction in error, we require ||q|| <= 1. We can construct such a q by
// shifting and scaling Chebyshev polynomials. Suppose the spectrum of A (or at
// least the part of the spectrum we care about) is contained within the
// interval [a, b], where a and b are of the same sign or zero. Let
//
//     s(z) = -1 + 2 (z-a) / (b-a).
//
// Then, for Chebyshev polynomial T_k(z),
//
//     ||T_k(s(z))|| <= 1 on [a, b]
//
// Finally, Let q(z) = T_k(s(z)) / T_k(s(0)). Since s(0) is not in [-1, 1],
// ||T_k(s(z))|| >= 1, so ||q(z)|| <= 1 and q(0) = 1. We can then back out p(z),
//
//     p(z) = (q(z) - 1) / z,
//
// which is then used to construct our improved guess.

namespace impl {

// Coefficients resulting from shifting and scaling Chebyshev polynomials are
// polynomial in a and b such that p(a, b) = \sum_{i=0}^{n-1} c_i a^i b^{n-1-j}.
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
 * I don't know a general formula for these coefficients, but:
 * denominator = polynomial((2n)!/(2n)!, (2n)!/(2!*(2n-2)!), ...,
 *                          (2n)!/((2k)!*(2n-2k)!), ..., (2n)!/(2n)!)(a, b).
 */

template <>
struct chebw<1> {
	// p_1(z) = 2 / (a + b)
	static constexpr polynomial p{2};
	static constexpr polynomial q{1, 1};
	double weights[1];

	constexpr chebw(double a, double b) :
		weights {p(a, b) / q(a, b)} {}
};

template <>
struct chebw<2> {
	// p_2(z) = (-8z + (a+b)) / (a^2 + 6ab + b^2)
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
	// p_3(z) = (32z^2 - 48(a + b)z + (18a^2 + 60ab + 18b^2)) /
	//          (a^3 + 15a^2b + 15ab^2 + b^3)
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
	// p_4(z) = (-128z^3 - 256(a+b)z^2 - (160a^2 + 448ab + 160b^2)z +
	//          (32a^3 + 224a^2b + 224ab^2 + 32b^3)) /
	//          (a^4 + 28a^3b + 70a^2b^2 + 28ab^3 + b^4)
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

// Evaluates a polynomial p_k(z) for matrix-valued z with
// coefficients defined in cheb<k> using Horner's method.
class chebyshev {
private:
	impl::chebw<4> chebw; // k = 4

	auto
	polynomial(vector r) const
	{
		const auto& weights = chebw.weights;
		static constexpr auto num_weights = sizeof(weights) / sizeof(double);

		auto s = weights[0] * r;
		for (int i = 1; i < num_weights; ++i) {
			vector t = r;
			gemv(-1.0, m, s, weights[i], t);
			//   ^~~~ the alternating sign of coefficients is hard-coded
			swap(s, t);
		}
		return s;
	}
protected:
	matrix m;
public:
	auto operator()(vector r) const { return polynomial(std::move(r)); }
	chebyshev(double a, double b, matrix m) : chebw(a, b), m(std::move(m)) {}
};

} // namespace algo
