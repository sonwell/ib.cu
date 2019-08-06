#pragma once
#include <cmath>
#include <algorithm>
#include <utility>
#include <functional>

namespace algo {
namespace impl {

template <typename arg_type, typename result_type>
constexpr auto
linear(arg_type step, result_type fa, result_type fb)
{
	auto s = fb / fa;
	return std::pair{2 * step * s, 1 - s};
}

template <typename arg_type, typename result_type>
constexpr auto
interp(arg_type a, arg_type b, arg_type c, arg_type step, result_type fa, result_type fb, result_type fc)
{
	if (a == c)
		return linear(step, fa, fb);

	auto q = fa / fc;
	auto r = fb / fc;
	auto s = fb / fa;

	return std::pair{2 * s * (step * q * (q - r) - (b - a) * (r - 1)),
					 (q - 1) * (r - 1) * (s - 1)};
}

} // namespace impl

template <typename func_type>
constexpr auto
zeroin(const double& ax, const double& bx, func_type&& f)
{
	using result_type = decltype(f(ax));
	using limits = std::numeric_limits<result_type>;
	constexpr auto eps = limits::epsilon();

	double a = ax;
	double b = bx;
	auto c = a;
	auto fa = f(a);
	auto fb = f(b);
	auto fc = fa;

	while (true) {
		auto prev_step = b - a;

		if (std::fabs(fc) < std::fabs(fb)) {
			std::tie(a, fa) = std::pair{b, fb};
			std::tie(b, fb) = std::pair{c, fc};
			std::tie(c, fc) = std::pair{a, fa};
		}

		auto tol = 2 * eps * std::fabs(b);
		auto new_step = c - b;

		if (std::fabs(new_step) <= 2 * tol || !fb)
			return b;

		if (std::fabs(prev_step) >= tol && std::fabs(fa) > std::fabs(fb)) {
			auto [p, q] = impl::interp(a, b, c, new_step, fa, fb, fc);
			if (p > 0) q = -q;
			else       p = -p;

			auto bound = std::min(1.5 * new_step * q - std::fabs(tol * q),
								  std::fabs(prev_step * q));
			if (p < bound)
				new_step = p / q;
		}

		if (std::fabs(new_step) < 2 * tol)
			new_step = 2 * ((new_step > 0) - (new_step <= 0)) * tol;

		std::tie(a, fa) = std::pair{b, fb};
		b += new_step / 2; fb = f(b);
		if (fb * fc > 0)
			std::tie(c, fc) = std::pair{a, fa};
	}
}

} // namespace algo
