#pragma once
#include <array>
#include "util/functional.h"
#include "bases/geometry.h"
#include "types.h"

namespace forces {

using util::math::nchoosek;

template <int dims>
struct vector {
	std::array<double, dims> v;

	constexpr decltype(auto) operator[](int n) { return v[n]; }
	constexpr decltype(auto) operator[](int n) const { return v[n]; }

	template <typename func, typename ... arg_types>
	constexpr decltype(auto)
	modify(const func& f, arg_types&& ... args)
	{
		using namespace util::functional;
		map(f, v, std::forward<arg_types>(args)...);
		return *this;
	}

	constexpr decltype(auto) operator*=(double s) { return modify([&] (double& v) { v *= s; }); }
	constexpr decltype(auto) operator/=(double s) { return modify([&] (double& v) { v *= s; }); }
	constexpr decltype(auto) operator+=(const vector& w) { return modify([] (double& v, const double& w) { v += w; }, w); }
	constexpr decltype(auto) operator-=(const vector& w) { return modify([] (double& v, const double& w) { v -= w; }, w); }
	template <std::size_t n> constexpr decltype(auto) get() { return operator[](n); }
	template <std::size_t n> constexpr decltype(auto) get() const { return operator[](n); }
};

template <int dims>
constexpr vector<dims>
operator*(double s, vector<dims> v)
{
	v *= s;
	return v;
}

template <int dims>
constexpr vector<dims>
operator*(vector<dims> v, double s)
{
	v *= s;
	return v;
}

template <int dims>
constexpr vector<dims>
operator/(vector<dims> v, double s)
{
	v /= s;
	return v;
}

template <int dims>
constexpr vector<dims>
operator+(vector<dims> l, const vector<dims>& r)
{
	l += r;
	return l;
}

template <int dims>
constexpr vector<dims>
operator-(vector<dims> l, const vector<dims>& r)
{
	l -= r;
	return l;
}

}

namespace std {

template <int dims>
struct tuple_size<forces::vector<dims>> : std::integral_constant<int, dims> {};

template <int n, int dims>
struct tuple_element<n, forces::vector<dims>> { using type = double; };

}

namespace forces {

template <typename> struct loader;

template <int dims> struct position { vector<dims+1> x; };
template <int dims> struct tangents { std::array<vector<dims+1>, dims> t; };
template <int dims> struct seconds { std::array<vector<dims+1>, nchoosek(dims+1, 2)> tt; };
template <int dims> struct normal { vector<dims+1> n; };
struct measure { double s; };

template <int dims>
struct loader<vector<dims>> {
	const double* values;
	int m;

	constexpr auto
	operator[](int n) const
	{
		vector<dims> v;
		auto j = n % m;
		for (int i = 0; i < dims; ++i)
			v[i] = values[i * m + j];
		return v;
	}

	loader(const matrix& m) :
		values(m.values()),
		m(m.rows() * m.cols() / dims) {}
};

template <>
struct loader<double> {
	const double* values;
	int m;

	constexpr double
	operator[](int n) const
	{
		auto j = n % m;
		return values[j];
	}

	loader(const matrix& m) :
		values(m.values()),
		m(m.rows() * m.cols()) {}
};

template <int dims>
struct loader<position<dims>> {
	loader<vector<dims+1>> x;

	constexpr auto
	operator[](int i) const
	{
		return x[i];
	}

	loader(const bases::geometry<dims>& g) : x{g.position} {}
};

template <int dims>
struct loader<tangents<dims>> {
	std::array<loader<vector<dims+1>>, dims> t;

	constexpr tangents<dims>
	operator[](int n) const
	{
		using namespace util::functional;
		auto cons = [] (auto ... v) { return tangents<dims>{std::move(v)...}; };
		return apply(cons, map([&] (const auto& l) { return l[n]; }, t));
	}

	static auto
	expand(const bases::geometry<dims>& g)
	{
		using namespace util::functional;
		auto op =[] (const auto& ... v)
		{
			return std::array{loader<vector<dims+1>>{v}...};
		};
		return apply(op, g.tangents);
	}

	loader(const bases::geometry<dims>& g) : t{expand(g)} {}
};

template <int dims>
struct loader<seconds<dims>> {
	std::array<loader<vector<dims+1>>, nchoosek(dims+1, 2)> tt;

	constexpr seconds<dims>
	operator[](int n) const
	{
		using namespace util::functional;
		auto cons = [] (auto ... v) { return seconds<dims>{std::move(v)...}; };
		return apply(cons, map([&] (const auto& l) { return l[n]; }, tt));
	}

	static auto
	expand(const bases::geometry<dims>& g)
	{
		using namespace util::functional;
		auto op =[] (const auto& ... v)
		{
			return std::array{loader<vector<dims+1>>{v}...};
		};
		return apply(op, g.second_derivatives);
	}

	loader(const bases::geometry<dims>& g) : tt{expand(g)} {}
};

template <int dims>
struct loader<normal<dims>> {
	loader<vector<dims+1>> n;

	constexpr auto
	operator[](int i) const
	{
		return n[i];
	}

	loader(const bases::geometry<dims>& g) : n{g.normal} {}
};

template <>
struct loader<measure> {
	loader<double> s;

	constexpr auto
	operator[](int n) const
	{
		return s[n];
	}

	template <int dims>
	loader(const bases::geometry<dims>& g) : s(g.sigma) {}
};

template <typename T, std::size_t n, std::size_t i>
struct subscripts {
	const T& ref;
	std::array<int, i> subs;

	constexpr decltype(auto)
	operator[](int j) const
	{
		std::array<int, i+1> ns;
		for (int k = 0; k < i; ++k)
			ns[k] = subs[k];
		ns[i] = j;
		if constexpr (i+1 == n)
			return resolve(ref, ns);
		else
			return subscripts<T, n, i+1>{ref, ns};
	}
};


template <int dims>
struct metric {
	using tangents = tangents<dims>;
	using subscripts = subscripts<metric, 2, 1>;
	static constexpr auto n = nchoosek(dims+1, 2);
	std::array<double, n> g;

	constexpr auto
	compute(const tangents& t)
	{
		std::array<double, n> v;
		auto ev = [&] (int i, int j) { return algo::dot(t.t[i], t.t[j]); };
		for (int i = 0; i < dims; ++i)
			for (int j = i; j < dims; ++j)
				v[i * (2 * dims - i - 1) / 2 + j] = ev(i, j);
		return v;
	}

	constexpr subscripts operator[](int i) const { return {*this, {i}}; };
	constexpr metric(const tangents& t) : g{compute(t)} {}
};

template <int dims>
constexpr decltype(auto)
resolve(const metric<dims>& m, const std::array<int, 2>& subs)
{
	auto [i, j] = subs;
	if (i > j) std::swap(i, j);
	return m.g[i * (2 * dims - i - 1) / 2 + j];
}

template <int dims>
struct curvature {
	using seconds = seconds<dims>;
	using normal = normal<dims>;
	using subscripts = subscripts<curvature, 2, 1>;
	static constexpr auto n = nchoosek(dims+1, 2);
	std::array<double, n> b;

	constexpr auto
	compute(const seconds& s, const normal& m)
	{
		auto ev = [&] (int i) { return algo::dot(s.tt[i], m.n); };
		std::array<double, n> v;
		for (int i = 0; i < n; ++i)
			v[i] = ev(i);
		return v;
	}

	constexpr subscripts operator[](int i) const { return {*this, {i}}; };
	constexpr curvature(const seconds& s, const normal& n) : b{compute(s, n)} {}
	template <typename T> constexpr curvature(const T& t) :
		curvature{(const seconds&) t, (const normal&) t} {}
};

template <int dims>
constexpr decltype(auto)
resolve(const curvature<dims>& c, const std::array<int, 2>& subs)
{
	auto [i, j] = subs;
	if (i < j) std::swap(i, j);
	return c.b[i * (2 * dims - i - 1) / 2 + j];
}

template <int dims>
struct christoffel {
	using tangents = tangents<dims>;
	using seconds = seconds<dims>;
	static constexpr auto n = nchoosek(dims+1, 2);
	std::array<double, dims * n> g;

	constexpr auto
	compute(const tangents& t, const seconds& s)
	{
		auto ev = [&] (int i, int j) { return algo::dot(s.tt[i], t.t[j]); };
		std::array<double, dims * n> v;
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < n; ++i)
				v[k * n + i] = ev(i, k);
		return v;
	};

	constexpr auto operator[](int n) const { return subscripts<christoffel, 3, 1>{*this, {n}}; };
	constexpr christoffel(const tangents& t, const seconds& s) : g{compute(t, s)} {}
	template <typename T> constexpr christoffel(const T& t) :
		christoffel{(const tangents&) t, (const seconds&) t} {}
};

template <int dims>
constexpr decltype(auto)
resolve(const christoffel<dims>& c, const std::array<int, 3>& subs)
{
	auto [i, j, k] = subs;
	if (i > j) std::swap(i, j);
	return c.g[k * nchoosek(dims+1, 2) + i * (2* dims - i - 1) / 2 + j];
}

} // namespace forces
