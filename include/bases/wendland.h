#include <array>
#include "util/functional.h"
#include "util/math.h"
#include "algo/gcd.h"
#include "differentiation.h"

namespace bases {

template <unsigned int k>
struct polynomial {
	std::array<double, k+1> coeff;

	constexpr double
	operator()(double x) const
	{
		double v = coeff[k];
		for (int i = k-1; i >= 0; --i) {
			v = x * v + coeff[i];
		}
		return v;
	}

	constexpr double& operator[](std::size_t n) { return coeff[n]; }
	constexpr const double& operator[](std::size_t n) const { return coeff[n]; }

	constexpr polynomial(const double (&coeff)[k+1]) :
		coeff(coeff) {}

	constexpr polynomial(std::array<double, k+1> coeff) :
		coeff(coeff) {}

	template <typename ... value_types>
	constexpr polynomial(value_types ... values) :
		coeff{(double) values...} { static_assert(sizeof...(value_types) == k+1); }

	constexpr polynomial() : coeff{0} {}
};

template <unsigned int k, unsigned l>
constexpr auto
operator+(polynomial<k> p, polynomial<l> q)
{
	constexpr auto n = k > l ? l : k;
	constexpr auto m = k > l ? k : l;
	polynomial<m> r;
	for (auto i = 0u; i < n; ++i)
		r[i] = p[i] + q[i];
	return r;
}

template <unsigned int k, unsigned l>
constexpr auto
operator-(polynomial<k> p, polynomial<l> q)
{
	constexpr auto n = k > l ? l : k;
	constexpr auto m = k > l ? k : l;
	polynomial<m> r;
	for (auto i = 0u; i <= n; ++i)
		r[i] = p[i] - q[i];
	return r;
}

template <unsigned int k, unsigned l>
constexpr auto
operator*(polynomial<k> p, polynomial<l> q)
{
	constexpr auto m = k * l;
	polynomial<m> r;
	for (int i = 0u; i <= m; ++i)
		for (int j = 0; j <= i; ++i)
			if (j <= k && i - j <= l)
				r[i] += p[j] * q[i - j];
	return r;
}

template <unsigned k, typename scalar>
requires requires (scalar s) { s * 1.; }
constexpr auto
operator*(scalar s, polynomial<k> p)
{
	for (int i = 0; i <= k; ++i)
		p[i] *= s;
	return p;
}

template <unsigned k, typename scalar>
requires requires (scalar s) { 1. * s; }
constexpr auto
operator*(polynomial<k> p, scalar s)
{
	for (int i = 0; i <= k; ++i)
		p[i] *= s;
	return p;
}

template <unsigned k, typename scalar>
requires requires (scalar s) { 1. / s; }
constexpr auto
operator/(polynomial<k> p, scalar s)
{
	for (int i = 0; i <= k; ++i)
		p[i] /= s;
	return p;
}

template <unsigned l, unsigned k>
struct wendland : basic_function {
	static constexpr auto
	binomial()
	{
		polynomial<l> p;
		int c = 1;
		for (int i = 0; i <= l; ++i) {
			p[i] = c;
			c *= (i - l);
			c /= (i + 1);
		}
		return p;
	}

	template <unsigned n>
	static constexpr auto
	integrate(polynomial<n> p)
	{
		using util::math::abs;
		polynomial<n+2> q;
		int f = 2;
		double a = p[0];
		int sa = abs(a);
		q[2] = -a;

		for (int i = 1; i <= n; ++i) {
			double b = p[i];
			int sb = abs(b);
			q[i + 2] = -b;

			if (!b) continue;

			sa = algo::gcd((i+2) * sa, f * sb);
			f *= (i + 2);
			auto sc = algo::gcd(sa, f);
			sa /= sc;
			f /= sc;
		}
		// sa := 1 ?

		int t = 0;
		for (int i = 2; i <= n+2; ++i) {
			auto sc = algo::gcd((int) abs(q[i]), i);
			q[i] /= sc;
			q[i] *= f;
			q[i] /= (sa * (i / sc));
			t += q[i];
		}
		q[0] = -t;
		return q;
	}

	template <unsigned n>
	static constexpr auto
	differentiate(polynomial<n> p)
	{
		static_assert(n >= 2, "attempting to use a Wendland function "
				"that is not as smooth as needed. Increase the order "
				"of your Wendland function and try again.");
		polynomial<n-2> q;
		for (int i = 0; i <= n-2; ++i)
			q[i] = (i+2) * p[i+2];
		return q;
	}

	template <unsigned n>
	static constexpr auto
	normalize(polynomial<n> p)
	{
		return p / util::math::abs(p[0]);
	}

	template <unsigned n, typename op_type, typename arg_type>
	static constexpr auto
	repeat(op_type op, arg_type&& arg)
	{
		using namespace util::functional;
		using sequence = std::make_integer_sequence<int, n>;
		if constexpr (n == 0)
			return arg;
		else {
			auto m = [&] (auto) { return op; };
			auto f = map(m, sequence{});
			auto g = [] (auto x, auto f) { return f(std::move(x)); };
			return apply(partial(foldl, g, arg), f);
		}
	}
public:
	template <int ... ds>
	constexpr double
	operator()(double r, partials<ds...> = {}) const
	{
		constexpr auto up = [] (auto p) { return integrate(std::move(p)); };
		constexpr auto down = [] (auto p) { return differentiate(std::move(p)); };
		constexpr auto p = normalize(repeat<k>(up, binomial()));
		constexpr auto q = repeat<sizeof...(ds)>(down, p);
		return r < 1 ? q(r) : 0.0;
	}
};

} // namespace bases
