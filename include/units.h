#pragma once
#include <cstdint>
#include <ratio>
#include <ostream>
#include "util/functional.h"

namespace units {

using tmpl_type = std::intmax_t;

namespace detail {

constexpr auto
pow(double base, tmpl_type exp)
{
	if (exp == 0) return 1.0;
	if (exp == 1) return base;
	if (exp < 0) return 1 / pow(base, -exp);
	auto r = pow(base, exp >> 1);
	return pow(base, exp & 1) * r * r;
}

static constexpr auto length_scale = 1'000.     /* per cm */;
static constexpr auto mass_scale   = 1'000'000. /* per  g */;
static constexpr auto time_scale   = 1'000.     /* per  s */;

template <tmpl_type dist, tmpl_type mass, tmpl_type time>
inline constexpr auto scale =
		pow(length_scale, dist) * pow(mass_scale, mass) * pow(time_scale, time);

} // namespace detail

template <tmpl_type dist, tmpl_type mass, tmpl_type time>
struct unit {
private:
	static constexpr auto distance_exponent = dist;
	static constexpr auto mass_exponent = mass;
	static constexpr auto time_exponent = time;
	static constexpr auto scale = detail::scale<dist, mass, time>;

	double value;
public:
	constexpr operator double() const { return value; }
	constexpr unit(double v) : value(v) {}

	constexpr unit& operator*=(double s) { value *= s; return *this; }
	constexpr unit& operator*=(const unit<0, 0, 0>& u) { value *= u.value; return *this; }
	constexpr unit& operator/=(double s) { value /= s; return *this; }
	constexpr unit& operator/=(const unit<0, 0, 0>& u) { value /= u.value; return *this; }
	constexpr unit& operator+=(const unit& u) { value += u.value; return *this; }
	constexpr unit& operator-=(const unit& u) { value -= u.value; return *this; }
};

std::string
superscript(tmpl_type e)
{
	constexpr const char* superscripts[] = {
		"⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"
	};

	if (!e) return superscripts[0];
	if (e < 0) return "⁻" + superscript(-e);
	auto recurse = e > 10 ? superscript(e / 10) : "";
	return recurse + superscripts[e % 10];
}

template <tmpl_type dist, tmpl_type mass, tmpl_type time>
std::ostream&
operator<<(std::ostream& out, const unit<dist, mass, time>& u)
{
	using namespace util::functional;
	constexpr auto scale = detail::scale<dist, mass, time>;
	constexpr const char* symbols[] = {"g", "cm", "s"};
	constexpr tmpl_type counts[] = {mass, dist, time};
	double raw = u;
	out << raw / scale;

	bool printed = false;
	auto e = [&] (const char* symbol, tmpl_type count)
	{
		if (count) {
			if (printed) out << "·";
			out << symbol << (count != 1 ? superscript(count) : "");
			printed = true;
		}
	};
	map(e, symbols, counts);
	return out;
}


template <tmpl_type d, tmpl_type m, tmpl_type t>
constexpr auto
operator+(unit<d, m, t> u1, unit<d, m, t> u2)
{
	return u1 += u2;
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
constexpr auto
operator-(unit<d, m, t> u1, unit<d, m, t> u2)
{
	return u1 -= u2;
}

template <tmpl_type d1, tmpl_type m1, tmpl_type t1,
		 tmpl_type d2, tmpl_type m2, tmpl_type t2>
constexpr auto
operator*(unit<d1, m1, t1> u1, unit<d2, m2, t2> u2)
{
	return unit<d1+d2, m1+m2, t1+t2>{(double) u1 * (double) u2};
}

template <tmpl_type d1, tmpl_type m1, tmpl_type t1,
		 tmpl_type d2, tmpl_type m2, tmpl_type t2>
constexpr auto
operator/(unit<d1, m1, t1> u1, unit<d2, m2, t2> u2)
{
	return unit<d1-d2, m1-m2, t1-t2>{(double) u1 / (double) u2};
}

template <typename scalar, tmpl_type d, tmpl_type m, tmpl_type t,
		 typename = std::enable_if_t<std::is_arithmetic_v<scalar>>>
constexpr auto
operator*(scalar v, unit<d, m, t> u)
{
	return u *= v;
}

template <typename scalar, tmpl_type d, tmpl_type m, tmpl_type t,
		 typename = std::enable_if_t<std::is_arithmetic_v<scalar>>>
constexpr auto
operator*(unit<d, m, t> u, scalar v)
{
	return u *= v;
}

template <tmpl_type num, tmpl_type den, tmpl_type d, tmpl_type m, tmpl_type t>
constexpr auto
operator*(std::ratio<num, den>, unit<d, m, t> u)
{
	u *= num;
	u /= den;
	return u;
}

template <tmpl_type num, tmpl_type den, tmpl_type d, tmpl_type m, tmpl_type t>
constexpr auto
operator*(unit<d, m, t> u, std::ratio<num, den> r)
{
	return r * u;
}

template <typename scalar, tmpl_type d, tmpl_type m, tmpl_type t,
		 typename = std::enable_if_t<std::is_arithmetic_v<scalar>>>
constexpr auto
operator/(unit<d, m, t> u, scalar v)
{
	return u /= v;
}

template <typename scalar, tmpl_type d, tmpl_type m, tmpl_type t,
		 typename = std::enable_if_t<std::is_arithmetic_v<scalar>>>
constexpr auto
operator/(scalar v, unit<d, m, t> u)
{
	return unit<-d, -m, -t>{v / (double) u};
}

template <tmpl_type num, tmpl_type den, tmpl_type d, tmpl_type m, tmpl_type t>
constexpr auto
operator/(unit<d, m, t> u, std::ratio<num, den> r)
{
	u *= den;
	u /= num;
	return u;
}

using scalar = unit<0, 0, 0>;
using length = unit<1, 0, 0>;
using mass = unit<0, 1, 0>;
using time = unit<0, 0, 1>;
using force = unit<1, 1, -2>;
using velocity = unit<1, 0, -1>;
using acceleration = unit<1, 0, -2>;
using hertz = unit<0, 0, -1>;
using viscosity = unit<-1, 1, -1>;
using volume = unit<3, 0, 0>;
using area = unit<2, 0, 0>;
using diffusivity = unit<2, 0, -1>;
using density = unit<-3, 1, 0>;
using work = unit<2, 1, -2>;
using energy = work;

inline constexpr auto atto = std::atto{};
inline constexpr auto femto = std::femto{};
inline constexpr auto pico = std::pico{};
inline constexpr auto nano = std::nano{};
inline constexpr auto micro = std::micro{};
inline constexpr auto milli = std::milli{};
inline constexpr auto centi = std::centi{};
inline constexpr auto deci = std::deci{};
inline constexpr auto deca = std::deca{};
inline constexpr auto hecto = std::hecto{};
inline constexpr auto kilo = std::kilo{};
inline constexpr auto mega = std::mega{};
inline constexpr auto giga = std::giga{};
inline constexpr auto tera = std::tera{};
inline constexpr auto peta = std::peta{};
inline constexpr auto exa = std::exa{};

inline constexpr length m = 100 /* cm per m */ * detail::scale<1, 0, 0>;
inline constexpr mass   g = detail::scale<0, 1, 0>;
inline constexpr time   s = detail::scale<0, 0, 1>;

inline constexpr auto kg = kilo * g;
inline constexpr auto N = kg * m / (s * s);
inline constexpr auto L = std::ratio<1, 1000>{} * m * m * m;
inline constexpr auto mL = milli * L;
inline constexpr auto P = std::ratio<1, 10>{} * kg / (m * s);
inline constexpr auto J = N * m;

namespace literals {

using ull = unsigned long long;
using lf = long double;
constexpr auto operator ""_m   (lf  v) { return v * m; }
constexpr auto operator ""_kg  (lf  v) { return v * kg; }
constexpr auto operator ""_s   (lf  v) { return v * s; }
constexpr auto operator ""_N   (lf  v) { return v * N; }
constexpr auto operator ""_cm  (lf  v) { return v * (centi * m); }
constexpr auto operator ""_mm  (lf  v) { return v * (milli * m); }
constexpr auto operator ""_um  (lf  v) { return v * (micro * m); }
constexpr auto operator ""_L   (lf  v) { return v * L; }
constexpr auto operator ""_mL  (lf  v) { return v * mL; }
constexpr auto operator ""_g   (lf  v) { return v * g; }
constexpr auto operator ""_ms  (lf  v) { return v * (milli * s); }
constexpr auto operator ""_us  (lf  v) { return v * (micro * s); }
constexpr auto operator ""_dyn (lf  v) { return v * 1e-5 * N; }
constexpr auto operator ""_P   (lf  v) { return v * P; }
constexpr auto operator ""_cP  (lf  v) { return v * (centi * P); }
constexpr auto operator ""_J   (lf  v) { return v * J; }
constexpr auto operator ""_erg (lf  v) { return v * 1e-7 * J; }

constexpr auto operator ""_m   (ull v) { return v * m; }
constexpr auto operator ""_kg  (ull v) { return v * kg; }
constexpr auto operator ""_s   (ull v) { return v * s; }
constexpr auto operator ""_N   (ull v) { return v * N; }
constexpr auto operator ""_cm  (ull v) { return v * (centi * m); }
constexpr auto operator ""_mm  (ull v) { return v * (milli * m); }
constexpr auto operator ""_um  (ull v) { return v * (micro * m); }
constexpr auto operator ""_L   (ull v) { return v * L; }
constexpr auto operator ""_mL  (ull v) { return v * (milli * L); }
constexpr auto operator ""_g   (ull v) { return v * g; }
constexpr auto operator ""_ms  (ull v) { return v * (milli * s); }
constexpr auto operator ""_us  (ull v) { return v * (micro * s); }
constexpr auto operator ""_dyn (ull v) { return v * 1e-5 * N; }
constexpr auto operator ""_P   (ull v) { return v * P; }
constexpr auto operator ""_cP  (ull v) { return v * (centi * P); }
constexpr auto operator ""_J   (ull v) { return v * J; }
constexpr auto operator ""_erg (ull v) { return v * 1e-7 * J; }

} // namespace literals
} // namespace units

using namespace units::literals;
