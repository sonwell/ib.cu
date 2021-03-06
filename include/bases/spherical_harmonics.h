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
public:
	static __host__ __device__ auto
	coefficient(unsigned l, unsigned m)
	{
		double c = 2.0 * l + 1.0;
		for (int i = l - m + 1; i <= l + m; ++i)
			c /= i;
		return sqrt(c);
	}

	template <unsigned l, unsigned m, int d>
	static __host__ __device__ auto
	associated(double t)
	{
		// Associated Legendre function P_{lm}(cos(t)), normalized to give the
		// simplest formulae.
		//
		// t is the angle of elevation measured from the north pole.

		if constexpr (l < m)
			return 0.0;

		// Recursive differentiated formula from Bosch, W. "On the Computation
		// of Derivatives of Legendre Functions." (2000).
		if constexpr (d != 0) {
			if constexpr (m == 0)
				return -sqrt(l * (l+1)) * associated<l, 1, d-1>(t);
			else
				return (sqrt((l+m)*(l-m+1)) * associated<l, m-1, d-1>(t) -
				        sqrt((l+m+1)*(l-m)) * associated<l, m+1, d-1>(t)) / 2;
		}

		// Undifferentiated formula from Green, R. "Spherical Harmonic Lighting:
		// The Gritty Details." (2003)
		double pmm = coefficient(l, m);
		double st = sin(t);
		for (unsigned i = 0; i < m; ++i)
			pmm *= st * (2*(m-i)-1);
		if constexpr (l == m)
			return pmm;
		double ct = cos(t);
		double pm1m = (2*m+1) * ct * pmm;
		for (int i = m+1; i < l; ++i) {
			double pm2m = ((2*i+1)*ct*pm1m-(i+m)*pmm)/(i-m+1);
			pmm = pm1m;
			pm1m = pm2m;
		}
		return pm1m;
	}

	template <unsigned l, int m, int ... ds>
	static __host__ __device__ auto
	harmonic(double t, double p, partials<ds...>)
	{
		using util::math::pow;
		using seq = std::make_integer_sequence<int, 2>;
		constexpr typename detail::counter<seq, ds...>::type counts;
		constexpr auto pi = M_PI;
		constexpr auto pi_2 = M_PI_2;
		constexpr unsigned n = m < 0 ? -m : m;
		constexpr auto d0 = util::get<0>(counts);
		constexpr auto d1 = util::get<1>(counts);
		constexpr int sign = (1 - ((m & 1) << 1));
		double w = associated<l, n, d1>(pi_2 + p);
		double c = 1.0 / (4 * pi);
		for (int i = -n+1; i < n+1; ++i)
			c /= (l + i);
		double v = sign * sqrt(c) * w;
		double ct = cos(n * t);
		double st = sin(n * t);
		double cyc[] = {st, ct, -st, -ct};
		return v * pow((double) n, d0) * cyc[(1 - (m < 0) + d0) % 4];
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
