#pragma once
#include <cmath>
#include "util/sequences.h"
#include "algo/lu.h"
#include "bases/traits.h"
#include "bases/fill.h"
#include "bases/polynomials.h"
#include "bases/differentiation.h"
#include "bases/closed_surface.h"

namespace bases {
namespace shapes {

struct sphere : closed_surface<2> {
private:
	static constexpr bases::traits<sphere> traits;
	using base = closed_surface<2>;
public:
	struct metric : differentiable {
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
protected:
	using base::weights;
	using params = double[2];
	static constexpr auto pi = M_PI;
	static constexpr metric d{};
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
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z = sin(p);

			xdata[0 * rows + tid] = x;
			xdata[1 * rows + tid] = y;
			xdata[2 * rows + tid] = z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <typename rbf>
	static vector
	weights(const matrix& x, rbf phi)
	{
		static constexpr auto weight =
			[] __device__ (const params& x) { return cos(x[1]) / (4 * pi); };
		return base::weights(x, phi, weight);
	}

	template <typename trait_type, typename basic, typename poly = polynomials<0>>
	sphere(int nd, int ns, bases::traits<trait_type> traits, basic phi, poly p = {}) :
		base(nd, ns, traits, phi, d, p) {}

	template <typename basic, typename poly = polynomials<0>>
	sphere(int nd, int ns, basic phi, poly p = {}) :
		sphere(nd, ns, traits, phi, p) {}
};

} // namespace shapes
} // namespace bases
