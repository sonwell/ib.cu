#pragma once
#include <cmath>
#include <fstream>
#include <filesystem>
#include "util/sequences.h"
#include "bases/traits.h"
#include "bases/fill.h"
#include "bases/spherical_harmonics.h"
#include "bases/differentiation.h"
#include "bases/closed_surface.h"
#include "bases/scaled.h"

namespace bases {
namespace shapes {

using namespace bases::meta;

struct spherical_metric : bases::metric {
private:
	template <std::size_t n, int ... cnts>
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
	template <std::size_t n, int ... ds>
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
	using metric = spherical_metric;
protected:
	using base::shape;
	using base::weights;
	using params = double[2];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};

	template <meta::traits traits, meta::basic interp,
	          meta::basic eval, meta::polynomial poly>
	sphere(int nd, int ns, traits tr, interp phi, eval psi, poly p) :
		base(nd, ns, tr, phi, bases::scaled{psi, 1.0, phi(2) / psi(2)}, d, p) {}

	template <meta::traits traits, meta::basic basic, meta::polynomial poly>
	sphere(int nd, int ns, traits tr, basic phi, poly p) :
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
		static std::filesystem::path root = "data";
		std::stringstream ss; ss << "sphere.w." << x.rows() << ".bin";
		if (std::filesystem::exists(root / ss.str())) {
			vector w;
			std::fstream f(root / ss.str(), std::ios::in | std::ios::binary);
			f >> linalg::io::binary >> w;
			return w;
		}

		static constexpr auto weight =
			[] __device__ (const params& x) { return cos(x[1]) / (4 * pi); };
		return base::weights(x, d, weight);
	}

	template <meta::basic interp, meta::basic eval,
	          meta::polynomial poly = spherical_harmonics<1>>
	sphere(int nd, int ns, interp phi, eval psi, poly p = {}) :
		sphere(nd, ns, traits, phi, psi, p) {}

	template <meta::basic basic, meta::polynomial poly = spherical_harmonics<1>>
	sphere(int nd, int ns, basic phi, poly p = {}) :
		sphere(nd, ns, phi, phi, p) {}
};

struct circle : public closed_surface<1> {
private:
	static constexpr bases::traits<circle> traits;
	using base = closed_surface<1>;
public:
	using metric = spherical_metric;
protected:
	using base::shape;
	using base::weights;
	using params = double[1];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};

	template <meta::traits traits, meta::basic interp, meta::basic eval, meta::polynomial poly>
	circle(int nd, int ns, traits tr, interp phi, eval psi, poly p) :
		base(nd, ns, tr, phi, bases::scaled{psi, 1.0, phi(2) / psi(2)}, d, p) {}

	template <meta::traits traits, meta::basic basic, meta::polynomial poly>
	circle(int nd, int ns, traits tr, basic phi, poly p) :
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

	template <meta::basic interp, meta::basic eval,
	          meta::polynomial poly = polynomials<0>>
	circle(int nd, int ns, interp phi, eval psi, poly p = {}) :
		circle(nd, ns, traits, phi, psi, p) {}

	template <meta::basic basic, meta::polynomial poly = polynomials<0>>
	circle(int nd, int ns, basic phi, poly p = {}) :
		circle(nd, ns, phi, phi, p) {}
};

} // namespace shapes
} // namespace bases
