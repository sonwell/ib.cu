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

/*
struct cosine_delta {
private:
	static constexpr auto pi2 = M_PI_2;
public:
	static constexpr auto meshwidths = 3;

	constexpr auto
	operator()(double r) const
	{
		using util::math::cos;
		return 0.25 * (1 + cos(pi2 * r));
	}
};

struct roma_delta {
	static constexpr auto meshwidths = 3;

	constexpr auto
	operator()(double r) const
	{
		using util::math::abs;
		using util::math::sqrt;
		using util::math::floor;
		constexpr double weights[2][2] = {{3.0, -1.0}, {0.0,  2.0}};
		auto s = abs(r);
		int t = floor(s + 0.5);
		auto d = t - s;
		return (2.0 + weights[t][0] * d + weights[t][1] * sqrt(1-3*d*d)) / 6.;
	}

};*/

} // namespace detail
} // namespace delta
} // namespace ib
