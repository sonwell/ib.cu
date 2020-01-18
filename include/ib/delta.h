#pragma once
#include <array>

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

struct standard_pattern {
	constexpr auto operator[](int n) const { return n; }
	constexpr standard_pattern(std::size_t, std::size_t) {}
};

} // namespace detail
} // namespace delta
} // namespace ib
