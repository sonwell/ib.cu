#pragma once
#include <array>
#include "util/math.h"
#include "util/functional.h"
#include "differentiation.h"
#include "polynomials.h"

namespace bases {

template <unsigned degree>
class spherical_harmonics : differentiable {
	static constexpr auto np = (degree+1) * (degree+1);
private:
	static constexpr auto
	coefficient(unsigned l, unsigned m)
	{
		double c = (2.0 - (m == 0)) * (2 * l + 1);
		for (int i = l - m + 1; i <= l + m; ++i)
			c /= i;
		return c;
	}

	template <unsigned l, unsigned m, int d>
	static __host__ __device__ auto
	associated(double t)
	{
		// Normalized associated Legendre function P_{lm}
		if constexpr (l < m)
			return 0.0;
		// Recursive differentiated formula from Bosch, W. "On the Computation
		// of Derivatives of Legendre Functions." (2000).
		if constexpr (d != 0) {
			if constexpr (m == 0)
				return - sqrt(l * (l+1) / 2) * associated<l, 1, d-1>(t);
			else if constexpr (m == l) {
				if constexpr (m == 1)
					return associated<1, 0, d-1>(t);
				return sqrt(l / 2) * associated<l, l-1, d-1>(t);
			}
			else
				return (sqrt((l+m)*(l-m+1)) * associated<l, m-1, d-1>(t) -
				        sqrt((l+m+1)*(l-m)) * associated<l, m+1, d-1>(t)) / 2;
		}
		// Undifferentiated formula from Green, R. "Spherical Harmonic Lighting:
		// The Gritty Details." (2003)
		double pmm = sqrt(coefficient(l, m));
		double ct = cos(t);
		for (unsigned i = 0; i < m; ++i)
			pmm *= ct * (2*(m-i)-1);
		if constexpr (l == m)
			return pmm;
		double st = sin(t);
		double pm1m = (2*m+1) * st * pmm;
		if constexpr (l == m+1)
			return pm1m;
		double pm2m;
		for (int i = m+1; i < l; ++i) {
			pm2m = ((2*i+1)*st*pm1m-(i+m)*pmm)/(i-m+1);
			pmm = pm1m;
			pm1m = pm2m;
		}
		return pm2m;
	}

	template <unsigned l, int m, int ... ds>
	static __host__ __device__ auto
	harmonic(double t, double p, partials<ds...>)
	{
		using util::math::pow;
		using seq = std::make_integer_sequence<int, 2>;
		constexpr typename detail::counter<seq, ds...>::type counts;
		constexpr auto pi = M_PI;
		constexpr unsigned n = m < 0 ? -m : m;
		constexpr auto d0 = util::get<0>(counts);
		constexpr auto d1 = util::get<1>(counts);
		double w = associated<l, n, d0>(p);
		double v = pow((double) n, d1) * w / (4 * pi);
		double ct = cos(n * t);
		double st = sin(n * t);
		double cyc[] = {st, ct, -st, -ct};
		if constexpr (m < 0)
			return v * cyc[(0 + d1)%4];
		else
			return v * cyc[(1 + d1)%4];
	}

	template <unsigned l, int ... ds>
	static __host__ __device__ auto
	expand(double t, double p, partials<ds...> d)
	{
		using namespace util::functional;
		std::array<double, 2*l+1> v;
		auto k = [&] (auto i)
		{
			constexpr int m = i - l;
			v[i] = spherical_harmonics::harmonic<l, m>(t, p, d);
		};
		map(k, std::make_integer_sequence<int, 2*l+1>{});
		return v;
	}

	template <int ... ds>
	static __host__ __device__ auto
	eval(const double (&p)[2], partials<ds...> d)
	{
		using namespace util::functional;
		std::array<double, np> v;
		auto k = [&] (auto i)
		{
			constexpr int l = i;
			auto w = expand<l>(p[0], p[1], d);
			for (int j = 0; j < 2*l+1; ++j)
				v[l*l + j] = w[j];
		};
		map(k, std::make_integer_sequence<int, degree+1>{});
		return v;
	}

public:
	template <int ... ds>
	__host__ __device__ auto
	operator()(const double (&x)[2], partials<ds...> p = partials<>()) const
	{
		return eval(x, p);
	}
};

template <unsigned degree>
struct is_polynomial_basis<spherical_harmonics<degree>> : std::true_type {};

} // namespace bases
