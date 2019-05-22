#pragma once
#include <cstdint>
#include <ratio>
#include <ostream>

namespace units {

using tmpl_type = std::intmax_t;

template <tmpl_type dist, tmpl_type mass, tmpl_type time>
struct unit {
	static constexpr auto distance_exponent = dist;
	static constexpr auto mass_exponent = mass;
	static constexpr auto time_exponent = time;

	__host__ __device__ constexpr operator double() const { return value; }
	__host__ __device__ constexpr unit(double v) : value(v) {}

	__host__ __device__ constexpr unit& operator*=(double s) { value *= s; return *this; }
	__host__ __device__ constexpr unit& operator*=(const unit<0, 0, 0>& u) { value *= u.value; return *this; }
	__host__ __device__ constexpr unit& operator/=(double s) { value /= s; return *this; }
	__host__ __device__ constexpr unit& operator/=(const unit<0, 0, 0>& u) { value /= u.value; return *this; }
	__host__ __device__ constexpr unit& operator+=(const unit& u) { value += u.value; return *this; }
	__host__ __device__ constexpr unit& operator-=(const unit& u) { value -= u.value; return *this; }
private:
	double value;
};

auto&
print_char(std::ostream& out, tmpl_type exp)
{
	if (!exp) return out;
	print_char(out, exp / 10);
	switch (exp % 10) {
		case 0 : return out << "⁰";
		case 1 : return out << "¹";
		case 2 : return out << "²";
		case 3 : return out << "³";
		case 4 : return out << "⁴";
		case 5 : return out << "⁵";
		case 6 : return out << "⁶";
		case 7 : return out << "⁷";
		case 8 : return out << "⁸";
		case 9 : return out << "⁹";
	}
	return out;
}

template <tmpl_type dist, tmpl_type mass, tmpl_type time>
std::ostream&
operator<<(std::ostream& out, const unit<dist, mass, time>& u)
{

	auto e = [&] (tmpl_type exp)
	{
		if (exp == 1) return;
		if (exp < 0) {
			out << "⁻";
			exp = -exp;
		}
		print_char(out, exp);
	};

	out << (double) u;
	if constexpr (!!dist) {
		out << 'm';
		e(dist);
	}

	if constexpr (!!mass) {
		if constexpr (!!dist)
			out << "·";
		out << "kg";
		e(mass);
	}

	if constexpr (!!time) {
		if constexpr (!!dist || !!mass)
			out << "·";
		out << "s";
		e(time);
	}
	return out;
}


template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator+(unit<d, m, t> u1, unit<d, m, t> u2)
{
	return u1 += u2;
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator-(unit<d, m, t> u1, unit<d, m, t> u2)
{
	return u1 -= u2;
}

template <tmpl_type d1, tmpl_type m1, tmpl_type t1,
		 tmpl_type d2, tmpl_type m2, tmpl_type t2>
__host__ __device__ constexpr auto
operator*(unit<d1, m1, t1> u1, unit<d2, m2, t2> u2)
{
	return unit<d1+d2, m1+m2, t1+t2>{(double) u1 * (double) u2};
}

template <tmpl_type d1, tmpl_type m1, tmpl_type t1,
		 tmpl_type d2, tmpl_type m2, tmpl_type t2>
__host__ __device__ constexpr auto
operator/(unit<d1, m1, t1> u1, unit<d2, m2, t2> u2)
{
	return unit<d1-d2, m1-m2, t1-t2>{(double) u1 / (double) u2};
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator*(double v, unit<d, m, t> u)
{
	return u *= v;
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator*(unit<d, m, t> u, double v)
{
	return u *= v;
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator/(unit<d, m, t> u, double v)
{
	return u /= v;
}

template <tmpl_type d, tmpl_type m, tmpl_type t>
__host__ __device__ constexpr auto
operator/(double v, unit<d, m, t> u)
{
	return unit<-d, -m, -t>{v / (double) u};
}

using scalar = unit<0, 0, 0>;
using distance = unit<1, 0, 0>;
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

namespace literals {

using ull = unsigned long long;
using lf = long double;

__host__ __device__ constexpr auto operator ""_m   (lf  v) { return distance (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_kg  (lf  v) { return mass     (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_s   (lf  v) { return time     (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_N   (lf  v) { return force    (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_cm  (lf  v) { return distance (v * 1e-2); }
__host__ __device__ constexpr auto operator ""_mm  (lf  v) { return distance (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_um  (lf  v) { return distance (v * 1e-6); }
__host__ __device__ constexpr auto operator ""_g   (lf  v) { return mass     (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_ms  (lf  v) { return time     (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_us  (lf  v) { return time     (v * 1e-6); }
__host__ __device__ constexpr auto operator ""_dyn (lf  v) { return force    (v * 1e-5); }
__host__ __device__ constexpr auto operator ""_P   (lf  v) { return viscosity(v * 1e-1); }
__host__ __device__ constexpr auto operator ""_cP  (lf  v) { return viscosity(v * 1e-3); }

__host__ __device__ constexpr auto operator ""_m   (ull v) { return distance (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_kg  (ull v) { return mass     (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_s   (ull v) { return time     (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_N   (ull v) { return force    (v * 1e+0); }
__host__ __device__ constexpr auto operator ""_cm  (ull v) { return distance (v * 1e-2); }
__host__ __device__ constexpr auto operator ""_mm  (ull v) { return distance (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_um  (ull v) { return distance (v * 1e-6); }
__host__ __device__ constexpr auto operator ""_g   (ull v) { return mass     (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_ms  (ull v) { return time     (v * 1e-3); }
__host__ __device__ constexpr auto operator ""_us  (ull v) { return time     (v * 1e-6); }
__host__ __device__ constexpr auto operator ""_dyn (ull v) { return force    (v * 1e-5); }
__host__ __device__ constexpr auto operator ""_P   (ull v) { return viscosity(v * 1e-1); }
__host__ __device__ constexpr auto operator ""_cP  (ull v) { return viscosity(v * 1e-3); }

} // namespace literals
} // namespace units

using namespace units::literals;
