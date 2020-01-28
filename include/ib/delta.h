#pragma once
#include <array>
#include "types.h"

namespace ib {
namespace delta {

template <typename> struct traits;

namespace detail {

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

} // namespace delta
} // namespace ib
