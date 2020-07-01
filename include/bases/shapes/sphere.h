#pragma once
#include <cmath>
#include "util/sequences.h"
#include "algo/lu.h"
#include "bases/traits.h"
#include "bases/fill.h"
#include "bases/polynomials.h"
#include "bases/differentiation.h"
#include "bases/closed_surface.h"
#include "bases/scaled.h"

namespace bases {
namespace shapes {

template <std::size_t n>
struct spherical_metric : bases::metric {
private:
	template <int ... cnts>
	constexpr double
	eval(const double (&xs)[n], const double (&xd)[n],
			util::sequence<int, cnts...>) const
	{
		constexpr int counts[] = {cnts...};
		double t = 1.0;
		double w = 1.0;
		for (int i = 0; i < n; ++i) {
			auto cs = cos(xs[i]);
			auto ss = sin(xs[i]);
			double cyc[] = {ss, cs, -ss, -cs};
			auto c = counts[i];
			auto cw = cyc[(1+c) % 4];
			auto sw = cyc[(0+c) % 4];
			auto cv = cos(xd[i]);
			auto sv = sin(xd[i]);
			t = t * 0.5 * (cw * cv + cv * cw)
			  + w * 0.5 * (sw * sv + sv * sw);
			if (c) w = 0.0;
		}
		return t;
	}
public:
	template <int ... ds>
	constexpr double
	operator()(const double (&xs)[n], const double (&xd)[n],
			partials<ds...> p = partials<>()) const
	{
		using count = typename partials<ds...>::template counts<n>;
		auto tmp = eval(xs, xd, count{});
		if constexpr (sizeof...(ds) > 0)
			return -tmp;
		else
			return sqrt(2.0 * max(1.0-tmp, 0.0));
	}
};

struct sphere : closed_surface<2> {
private:
	static constexpr bases::traits<sphere> traits;
	using base = closed_surface<2>;
public:
	using metric = spherical_metric<2>;
protected:
	using base::shape;
	using base::weights;
	using params = double[2];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};

	template <typename traits_type, typename interp, typename eval, typename poly,
			 typename = std::enable_if_t<bases::is_basic_function_v<interp>>,
			 typename = std::enable_if_t<bases::is_basic_function_v<eval>>,
			 typename = std::enable_if_t<bases::is_polynomial_basis_v<poly>>>
	sphere(int nd, int ns, bases::traits<traits_type> tr, interp phi, eval psi, poly p) :
		base(nd, ns, tr, phi, bases::scaled{psi, 1.0, phi(2) / psi(2)}, d, p) {}

	template <typename traits_type, typename basic, typename poly,
			 typename = std::enable_if_t<bases::is_basic_function_v<basic>>,
			 typename = std::enable_if_t<bases::is_polynomial_basis_v<poly>>>
	sphere(int nd, int ns, bases::traits<traits_type> tr, basic phi, poly p) :
		sphere(nd, ns, tr, phi, phi, p) {}
public:
	static matrix
	sample(int n)
	{
		static constexpr auto pi = M_PI;
		// Bauer spiral
		matrix params{n, 2};
		auto* values = params.values();

		auto k = [=] __device__ (int tid)
		{
			auto l = sqrt(pi * n);
			auto z = -1.0 + (2.0 * tid + 1.0) / n;
			auto phi = asin(z);
			auto theta = l * phi;
			values[0 * n + tid] = fmod(theta, 2 * pi);
			values[1 * n + tid] = phi;
		};
		util::transform<128, 7>(k, n);
		return params;
	}

	static matrix
	shape(const matrix& params)
	{
		using point = std::array<double, 3>;
		auto k = [] __device__ (auto params) -> point
		{
			auto [t, p] = params;
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z = sin(p);
			return {x, y, z};
		};
		return base::shape(params, k);
	}

	static vector
	weights(const matrix& x)
	{
		static constexpr auto weight =
			[] __device__ (const params& x) { return cos(x[1]) / (4 * pi); };
		return base::weights(x, d, weight);
	}

	template <typename interp, typename eval, typename poly = polynomials<0>,
	          typename = std::enable_if_t<bases::is_basic_function_v<interp> &&
	                                      bases::is_basic_function_v<eval>   &&
	                                      bases::is_polynomial_basis_v<poly>>>
	sphere(int nd, int ns, interp phi, eval psi, poly p = {}) :
		sphere(nd, ns, traits, phi, psi, p) {}

	template <typename basic, typename poly = polynomials<0>>
	sphere(int nd, int ns, basic phi, poly p = {}) :
		sphere(nd, ns, phi, phi, p) {}
};

struct circle : public closed_surface<1> {
private:
	static constexpr bases::traits<circle> traits;
	using base = closed_surface<1>;
public:
	using metric = spherical_metric<1>;
protected:
	using base::shape;
	using base::weights;
	using params = double[1];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};

	template <typename traits_type, typename interp, typename eval, typename poly,
	          typename = std::enable_if_t<bases::is_basic_function_v<interp> &&
	                                      bases::is_basic_function_v<eval>   &&
	                                      bases::is_polynomial_basis_v<poly>>>
	circle(int nd, int ns, bases::traits<traits_type> tr, interp phi, eval psi, poly p) :
		base(nd, ns, tr, phi, bases::scaled{psi, 1.0, phi(2) / psi(2)}, d, p) {}

	template <typename traits_type, typename basic, typename poly>
	circle(int nd, int ns, bases::traits<traits_type> tr, basic phi, poly p) :
		circle(nd, ns, tr, phi, phi, p) {}
public:
	static matrix
	sample(int n)
	{
		static constexpr auto pi = M_PI;
		matrix params{n, 1};
		auto* values = params.values();

		auto k = [=] __device__ (int tid)
		{
			auto z = (2.0 * tid + 1.0) / n;
			values[tid] = pi * z;
		};
		util::transform<128, 7>(k, n);
		return params;
	}

	static matrix
	shape(const matrix& params)
	{
		using point = std::array<double, 2>;
		auto k = [=] __device__ (auto x) -> point
		{
			auto [t] = x;
			return {cos(t), sin(t)};
		};
		return base::shape(params, k);
	}

	static vector
	weights(const matrix& x)
	{
		static constexpr auto weight =
			[] __device__ (const params& x) { return 1. / (2 * pi); };
		return base::weights(x, d, weight);
	}

	template <typename interp, typename eval, typename poly = polynomials<0>,
	          typename = std::enable_if_t<bases::is_basic_function_v<interp> &&
	                                      bases::is_basic_function_v<eval>   &&
	                                      bases::is_polynomial_basis_v<poly>>>
	circle(int nd, int ns, interp phi, eval psi, poly p = {}) :
		circle(nd, ns, traits, phi, psi, p) {}

	template <typename basic, typename poly = polynomials<0>>
	circle(int nd, int ns, basic phi, poly p = {}) :
		circle(nd, ns, phi, phi, p) {}
};

} // namespace shapes
} // namespace bases
