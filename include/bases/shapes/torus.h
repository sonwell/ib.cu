#pragma once
#include <cmath>
#include "util/sequences.h"
#include "bases/traits.h"
#include "bases/closed_surface.h"
#include "bases/differentiation.h"
#include "bases/scaled.h"

namespace bases {
namespace shapes {

using namespace bases::meta;

struct toroidal_metric : bases::metric {
private:
	template <typename ... arg_types>
	static constexpr bool
	all(arg_types ... args)
	{
		return (... && args);
	}
public:
	template <std::size_t n>
	constexpr double
	operator()(const double (&xs)[n], const double (&xd)[n],
			partials<> p = partials<>()) const
	{
		auto t = 0.0;
		for (int i = 0; i < n; ++i)
			t += 1 - cos(xs[i] - xd[i]);
		return sqrt(2 * t / n);
	}

	template <std::size_t n, int d, int ... ds>
	constexpr double
	operator()(const double (&xs)[n], const double (&xd)[n],
			partials<d, ds...> p) const
	{
		if constexpr (all(d == ds...)) {
			auto delta = xs[d] - xd[d];
			auto c = cos(delta);
			auto s = sin(delta);
			double cyc[] = {-c, s, c, -s};
			return cyc[(1 + sizeof...(ds)) % 4] / n;
		}
		else return 0.0;
	}
};

// This is the flat torus. It does not have a physically realizeable shape in
// R^3 with the metric above. The metric is the result of taking the Cartesian
// product of two circles, scaled.
struct torus : closed_surface<2> {
private:
	static constexpr bases::traits<torus> traits;
public:
	using metric = toroidal_metric;
protected:
	using closed_surface<2>::shape;
	using params = double[2];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};

	template <meta::traits traits, meta::basic interp, meta::basic eval,
	          meta::polynomial poly>
	torus(int nd, int ns, traits tr, interp phi, eval psi, poly p) :
		closed_surface(nd, ns, tr, phi, bases::scaled{psi, 1.0, phi(2) / psi(2)}, d, p) {}

	template <meta::traits traits, meta::basic basic, meta::polynomial poly>
	torus(int nd, int ns, traits tr, basic phi, poly p) :
		torus(nd, ns, tr, phi, phi, p) {}
public:
	static matrix
	sample(int n)
	{
		matrix params{n, 2};
		auto* values = params.values();

		auto k = [=] __device__ (int tid)
		{
			auto l = ceil(sqrt(n));
			auto z = (0.0 + tid) / n;
			auto theta = fmod(2 * pi * l * z, 2 * pi);
			auto phi = fmod(2 * pi * z, 2 * pi);
			values[0 * n + tid] = theta;
			values[1 * n + tid] = phi;
		};
		util::transform<128, 7>(k, n);
		return params;
	}

	static matrix
	shape(const matrix& params)
	{
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto x = (1 + cos(t)) * cos(p);
			auto y = (1 + cos(t)) * sin(p);
			auto z = sin(t);

			xdata[0 * rows + tid] = x;
			xdata[1 * rows + tid] = y;
			xdata[2 * rows + tid] = z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	static vector
	weights(const matrix& x)
	{
		auto n = x.rows();
		return vector{n, linalg::fill(4 * pi * pi / n)};
	}

	template <meta::basic interp, meta::basic eval,
	          meta::polynomial poly = polynomials<0>>
	torus(int nd, int ns, interp phi, eval psi, poly p = {}) :
		torus(nd, ns, traits, phi, psi, p) {}

	template <meta::basic basic, meta::polynomial poly = polynomials<0>>
	torus(int nd, int ns, basic phi, poly p = {}) :
		torus(nd, ns, traits, phi, p) {}
};

} // namespace shapes
} // namespace bases
