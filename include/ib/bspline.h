#pragma once
#include <cmath>
#include <array>
#include "delta.h"

namespace ib {
namespace delta {

/* Cardinal B-spline delta functions. See:
 *   * Lee. PhD Thesis, UNC Chapel Hill. 2020.
 *   * Griffith and Patankar. Ann Rev Fluid Mech 52(1):421-448. */

template <std::size_t n>
struct bspline {
	constexpr double
	operator()(double r) const
	{
		if constexpr (n == 0)
			return -0.5 <= r && r < 0.5;
		else {
			constexpr bspline<n-1> f;
			constexpr auto b = (n + 1.0) / (2.0 * n);
			auto a = r / n;
			return (b + a) * f(r+0.5) + (b - a) * f(r-0.5);
		}
	}
};

template <std::size_t n>
struct traits<bspline<n>> {
public:
	static constexpr auto calls = n+1;
	static constexpr auto meshwidths = n+1;
	using rule = detail::rule<0, calls>;
private:
	static constexpr auto
	get_offsets()
	{
		std::array<int, n+1> offsets = {};
		for (std::size_t i = 0; i <= n; ++i)
			offsets[i] = i;
		return offsets;
	}

	static constexpr auto
	get_rules()
	{
		std::array<double, n+1> v = {0};
		std::array<rule, n+1> rules = {};
		constexpr detail::polynomial<0> p = {};
		for (int i = 0; i < n+1; ++i) {
			v[i] = 1.0;
			rules[i] = rule{p, {v}};
			v[i] = 0.0;
		}
		return rules;
	};
public:
	static constexpr auto offsets = get_offsets();
	static constexpr auto rules = get_rules();
	template <std::size_t dimensions>
		using pattern = standard_pattern<dimensions, meshwidths>;
};

} // namespace delta
} // namespace ib
