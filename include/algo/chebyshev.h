#pragma once
#include <utility>
#include <cmath>
#include "types.h"
#include "gershgorin.h"
#include "preconditioner.h"

namespace algo {
namespace impl {

template <int n>
constexpr inline double
polytail(double a, double b, double w)
{
	return w * pow(b, n);
}

template <int n, typename ... arg_types>
constexpr inline double
polytail(double a, double b, double w, arg_types&& ... ws)
{
	constexpr auto m = sizeof...(arg_types);
	return w * pow(a, m) * pow(b, n - m) +
		polytail<n>(a, b, std::forward<arg_types>(ws)...);
}

template <typename ... arg_types>
constexpr inline double
polynomial(double a, double b, arg_types&& ... ws)
{
	return polytail<sizeof...(ws)-1>(a, b, std::forward<arg_types>(ws)...);
}

template <int degree> struct chebw;

/*
 * denominator = polynomial(a, b, (2n)!/(2n)!, (2n)!/(2!*(2n-2)!), ...,
 *                          (2n)!/((2k)!*(2n-2k)!), ..., (2n)!/(2n)!).
 */

template <>
struct chebw<1> {
	double denominator;
	double weights[1];

	constexpr chebw(double a, double b) :
		denominator(polynomial(a, b, 1, 1)),
		weights{
			polynomial(a, b, 2)
		} {}
};

template <>
struct chebw<2> {
	double denominator;
	double weights[2];

	constexpr chebw(double a, double b) :
		denominator(polynomial(a, b, 1, 6, 1)),
		weights{
			polynomial(a, b, 8),
			polynomial(a, b, 8, 8)
		} {}
};

template <>
struct chebw<3> {
	double denominator;
	double weights[3];

	constexpr chebw(double a, double b) :
		denominator(polynomial(a, b, 1, 15, 15, 1)),
		weights{
			polynomial(a, b, 32),
			polynomial(a, b, 48, 48),
			polynomial(a, b, 18, 60, 18)
		} {}
};

template <>
struct chebw<4> {
	double denominator;
	double weights[4];

	constexpr chebw(double a, double b) :
		denominator(polynomial(a, b, 1, 28, 70, 28, 1)),
		weights{
			polynomial(a, b, 128),
			polynomial(a, b, 256, 256),
			polynomial(a, b, 160, 448, 160),
			polynomial(a, b, 32, 224, 224, 32)
		} {}
};

} // namespace impl

class chebyshev {
private:
	impl::chebw<4> chebw;
protected:
	const matrix& m;

	vector
	polynomial(const vector& r) const
	{
		const auto& weights = chebw.weights;
		const auto& denominator = chebw.denominator;
		static constexpr auto num_weights = sizeof(weights) / sizeof(double);

		vector y = (weights[0] / denominator) * r;
		if constexpr (num_weights == 1)
			return y;

		vector z{size(r)};
		for (int i = 1; i < num_weights; ++i) {
			gemv(-1.0, m, y, 0.0, z);
			axpy(weights[i] / denominator, r, z);
			swap(y, z);
		}
		return y;
	}
public:
	vector
	operator()(const vector& b) const
	{
		return polynomial(b);
	}

	chebyshev(double a, double b, const matrix& m) :
		chebw(a, b), m(m) {}
	chebyshev(chebyshev&& o) :
		chebw(std::move(o.chebw)), m(o.m) {}
};

} // namespace algo
