#pragma once
#include <type_traits>
#include <iomanip>
#include <ostream>
#include <complex>
#include <cmath>

//#ifndef __CUDA__
//#define __host__
//#define __device__
//#define __forceinline__ inline
//#endif

namespace linalg {

template <typename base_type> class complex;

template <typename base_type>
	__host__ __device__ constexpr base_type real(const complex<base_type>&);
template <typename base_type>
	__host__ __device__ constexpr base_type imag(const complex<base_type>&);
template <typename base_type>
	__host__ __device__ constexpr base_type fabs(const complex<base_type>&);
template <typename base_type>
	__host__ __device__ constexpr complex<base_type> conj(const complex<base_type>&);

namespace detail {
	template <typename> struct complex_base { using type = double; };
	template <> struct complex_base<float> { using type = float; };
	template <typename base_type> using complex_base_t = typename complex_base<base_type>::type;
}

template <typename base_type>
class complex {
private:
	static_assert(std::is_same_v<base_type, float> ||
			std::is_same_v<base_type, double>,
			"complex only supports float and double types");
	base_type x;
	base_type y;

	template <typename real_type, typename imag_type>
	__host__ __device__ constexpr complex&
	assign(real_type r, imag_type i)
	{
		static_assert(std::is_arithmetic_v<real_type>,
				"complex can only be assigned arithmetic types");
		static_assert(std::is_arithmetic_v<imag_type>,
				"complex can only be assigned arithmetic types");
		real(r); imag(i);
		return *this;
	}
public:
	using value_type = base_type;

	constexpr operator std::complex<value_type>() const { return {real(), imag()}; }
	constexpr operator double() const { return real(); }
	constexpr operator float() const { return real(); }

	template <typename other_type>
	__host__ __device__ constexpr complex&
	operator=(other_type v) { return assign(v, 0.0); }

	template <typename other_base>
	__host__ __device__ constexpr complex&
	operator=(const complex<other_base>& z) { return assign(z.x, z.y); }

	__host__ __device__ constexpr value_type real() const { return x; }
	__host__ __device__ constexpr void real(value_type v) { x = v; }
	__host__ __device__ constexpr value_type imag() const { return y; }
	__host__ __device__ constexpr void imag(value_type v) { y = v; }

	template <typename other_type>
	__host__ __device__ constexpr complex&
	operator+=(other_type v) { return assign(x+v, y); }

	template <typename other_type>
	__host__ __device__ constexpr complex&
	operator-=(other_type v) { return assign(x-v, y); }

	template <typename other_type>
	__host__ __device__ constexpr complex&
	operator*=(other_type v) { return assign(x*v, y*v); }

	template <typename other_type>
	__host__ __device__ constexpr complex&
	operator/=(other_type v) { return assign(x/v, y/v); }

	template <typename other_base>
	__host__ __device__ constexpr complex&
	operator+=(const complex<other_base>& z) { return assign(x+z.x, y+z.y); }

	template <typename other_base>
	__host__ __device__ constexpr complex&
	operator-=(const complex<other_base>& z) { return assign(x-z.x, y-z.y); }

	template <typename other_base>
	__host__ __device__ constexpr complex&
	operator*=(complex<other_base> z) { return assign(x*z.x-y*z.y, x*z.y+y*z.x); }

	template <typename other_base>
	__host__ __device__ constexpr complex&
	operator/=(const complex<other_base>& z) { return operator*=(conj(z)/fabs(z)); }

	template <typename real_type, typename imag_type>
	__host__ __device__ constexpr complex(const real_type& re = 0, const imag_type& im = 0) :
		x{(value_type) re}, y{(value_type) im} {}

	template <typename other_base>
	__host__ __device__ constexpr complex(const complex<other_base>& z) :
		x{(value_type) z.real()}, y{(value_type) z.imag()} {}

	constexpr complex(const std::complex<value_type>& z) :
		x{z.real()}, y{z.imag()} {}
};

template <typename real_type, typename imag_type>
	complex(const real_type& re, const imag_type& im) ->
	complex<detail::complex_base_t<decltype(re + im)>>;

template <typename real_type>
	complex(real_type re) -> complex<detail::complex_base_t<real_type>>;

complex() -> complex<double>;

template <typename base_type>
__host__ __device__ __forceinline__ constexpr base_type
real(const complex<base_type>& z)
{ return z.real(); }

__host__ __device__ __forceinline__ constexpr double
real(double x) { return x; }

__host__ __device__ __forceinline__ constexpr float
real(float x) { return x; }

__host__ __device__ __forceinline__ constexpr double
imag(double) { return 0; }

__host__ __device__ __forceinline__ constexpr float
imag(float) { return 0; }

template <typename base_type>
__host__ __device__ __forceinline__ constexpr base_type
imag(const complex<base_type>& z)
{ return z.imag(); }

template <typename base_type>
__host__ __device__ __forceinline__ constexpr base_type
fabs(const complex<base_type>& z)
{
	auto re = z.real();
	auto im = z.imag();
	return re * re + im * im;
}

template <typename base_type>
__host__ __device__ __forceinline__ constexpr complex<base_type>
conj(const complex<base_type>& z)
{ return {z.real(), -z.imag()}; }

template <typename base_type, typename other_base>
__host__ __device__ constexpr auto
operator+(complex<base_type> z, const complex<other_base>& x)
{
	using result_type = decltype(real(z) + real(x));
	return complex<result_type>{z} += x;
}

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator+(complex<base_type> z, const other_type& x)
{ return z + complex{x}; }

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator+(const other_type& x, complex<base_type> z)
{ return complex{x} + z; }

template <typename base_type, typename other_base>
__host__ __device__ constexpr auto
operator-(complex<base_type> z, const complex<other_base>& x)
{
	using result_type = decltype(real(z) - real(x));
	return complex<result_type>{z} -= x;
}

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator-(complex<base_type> z, const other_type& x)
{ return z - complex{x}; }

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator-(const other_type& x, complex<base_type> z)
{ return complex{x} - z; }

template <typename base_type, typename other_base>
__host__ __device__ constexpr auto
operator*(complex<base_type> z, const complex<other_base>& x)
{
	using result_type = decltype(real(z) * real(x));
	return complex<result_type>{z} *= x;
}

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator*(complex<base_type> z, const other_type& x)
{ return z * complex{x}; }

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator*(const other_type& x, complex<base_type> z)
{ return complex{x} * z; }

template <typename base_type, typename other_base>
__host__ __device__ constexpr auto
operator/(complex<base_type> z, const complex<other_base>& x)
{
	using result_type = decltype(real(z) / real(x));
	return complex<result_type>{z} /= x;
}

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator/(complex<base_type> z, const other_type& x)
{ return z / complex{x}; }

template <typename base_type, typename other_type>
__host__ __device__ constexpr auto
operator/(const other_type& x, complex<base_type> z)
{ return complex{x} / z; }

template <typename base_type>
__host__ __device__ constexpr auto
operator+(complex<base_type> z) { return z; }

template <typename base_type>
__host__ __device__ constexpr auto
operator-(complex<base_type> z) { return -1 * z; }

template <typename base_type>
inline std::ostream&
operator<<(std::ostream& out, const complex<base_type>& z)
{
	auto re = real(z);
	auto im = imag(z);
	auto prec = out.precision();

	if (re) {
		out << std::setprecision(prec) << re;
		if (im) out << (im > 0 ? "+" : "") << std::setprecision(prec) << std::fabs(im) << 'i';
	}
	else if (im == 1) out << 'i';
	else if (im) out << std::setprecision(prec) << im << 'i';
	else out << std::setprecision(prec) << 0;

	return out;
}

constexpr complex<double> operator ""_i (long double v) { return {0, v}; }
constexpr complex<double> operator ""_i (unsigned long long v) { return {0, v}; }
constexpr complex<float> operator ""_if (long double v) { return {0, v}; }
constexpr complex<float> operator ""_if (unsigned long long v) { return {0, v}; }

} // namespace linalg

using linalg::operator ""_i;
using linalg::operator ""_if;
