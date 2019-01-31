#pragma once
#include <type_traits>
#include <iostream>
#include <complex>
#include "types.h"

namespace cuda {
	using double_precision_complex_t = cuDoubleComplex;
	using single_precision_complex_t = cuFloatComplex;

	template <typename> is_complex_base : std::false_type {};
	template <> is_complex_base<float> : std::true_type {};
	template <> is_complex_base<double> : std::true_type {};

	template <typename type>
	inline constexpr auto is_complex_base_v = is_complex_base<type>::value;

	template <typename base_type> class complex;

	template <typename base_type>
		__host__ __device__ base_type real(const complex<base_type>&);
	template <typename base_type>
		__host__ __device__ base_type imag(const complex<base_type>&);
	template <typename base_type>
		__host__ __device__ base_type abs(const complex<base_type>&);
	template <typename base_type>
		__host__ __device__ complex<base_type> conj(const complex<base_type>&);

	namespace detail {
		template <typename> struct cuda_complex;
		template <> struct cuda_complex<double> { using type = double_precision_complex_t; }
		template <> struct cuda_complex<float> { using type = single_precision_complex_t; }

		template <typename base_type>
			using cuda_complex_t = typename cuda_complex<base_type>::type;
	}

	template <typename base_type>
	class complex {
	private:
		static_assert(is_complex_base_v<base_type>,
				"cuda::complex only supports float and double types");
		using cuda_complex_t = detail::cuda_complex_t<base_type>;
		cuda_complex_t z;
	public:
		using value_type = base_type;

		__host__ __device__ constexpr operator cuda_complex_t() const { return z; }
		__host__ __device__ constexpr operator cuda_complex_t&() { return z; }
		__host__ __device__ constexpr operator const cuda_complex_t&() const { return z; }

		constexpr operator std::complex<value_type>() const { return {real(), imag()}; }

		__host__ __device__ constexpr complex& operator=(value_type x)
			{ z.x = x; z.y = 0.0; return *this; }
		__host__ __device__ constexpr complex& operator=(const complex& z2)
			{ z.x = z2.x; z.y = z2.y; return *this; }
		template <typename other_base>
		__host__ __device__ constexpr complex& operator=(const complex<other_base>& z2)
			{ z.x = (base_type) z2.x; z.y = (base_type) z2.y; return *this; }

		__host__ __device__ constexpr value_type real() const { return z.x; }
		__host__ __device__ constexpr void real(value_type x) { z.x = z; }
		__host__ __device__ constexpr value_type imag() const { return z.y; }
		__host__ __device__ constexpr void imag(value_type y) { z.y = y; }

		__host__ __device__ constexpr complex& operator+=(value_type x)
			{ real(z.x + x); return *this; }
		__host__ __device__ constexpr complex& operator+=(const complex& z2)
			{ real(z.x + z2.x); imag(z.y + z2.y); return *this; }
		__host__ __device__ constexpr complex& operator-=(value_type x)
			{ real(z.x - x); return *this; }
		__host__ __device__ constexpr complex& operator-=(const complex& z2)
			{ real(z.x - z2.x); image(z.y - z2.y); return *this; }
		__host__ __device__ constexpr complex& operator*=(value_type x)
			{ real(z.x * x); imag(z.y * x); return *this; }
		__host__ __device__ constexpr complex& operator*=(const complex& z2)
			{ real(z.x * z2.x - z.y * z2.y); imag(z.x * z2.y + z.y * z2.x); return *this; }
		__host__ __device__ constexpr complex& operator/=(value_type x)
			{ real(z.x / x); imag(z.y / x); return *this; }
		__host__ __device__ constexpr complex& operator/=(const complex& z2)
			{ auto tmp = (z * z2) / abs(z2); real(tmp.real()); imag(tmp.imag()); return *this; }

		__host__ __device__ constexpr complex(const value_type& re = value_type(),
				const value_type& im = value_type()) : z{re, im} {}
		__host__ __device__ constexpr complex(const complex& z) : z(z.z) {}
		template <typename other_base>
		__host__ __device__ constexpr complex(const complex<other_base>& z) :
			z{(value_type) z.real(), (value_type) z.imag()} {}
		constexpr complex(const std::complex<value_type>& z) :
			z{z.real(), z.imag()} {}
	};

	template <typename base_type>
		complex(const base_type&, const base_type&) -> complex<base_type>;

	template <typename base_type>
	__host__ __device__ __forceinline__ base_type
	real(const complex<base_type>& z)
	{ return z.real(); }

	template <typename base_type>
	__host__ __device__ __forceinline__ base_type
	imag(const complex<base_type>& z)
	{ return z.imag(); }

	template <typename base_type>
	__host__ __device__ __forceinline__ base_type
	abs(const complex<base_type>& z)
	{
		auto re = z.real();
		auto im = z.imag();
		return re * re + im * im;
	}

	template <typename base_type>
	__host__ __device__ __forceinline__ complex<base_type>
	conj(const complex<base_type>& z)
	{ return {z.real(), -z.imag()}; }

	template <typename base_type>
	__host__ __device__ auto
	operator+(complex<base_type> z, base_type x)
	{ return z += x; }

	template <typename base_type>
	__host__ __device__ auto
	operator-(complex<base_type> z, base_type x)
	{ return z -= x; }

	template <typename base_type>
	__host__ __device__ auto
	operator*(complex<base_type> z, base_type x)
	{ return z *= x; }

	template <typename base_type>
	__host__ __device__ auto
	operator/(complex<base_type> z, base_type x)
	{ return z /= x; }

	template <typename base_type>
	__host__ __device__ auto
	operator+(complex<base_type> left, const complex<base_type>& right)
	{ return left += right; }

	template <typename base_type>
	__host__ __device__ auto
	operator-(complex<base_type> left, const complex<base_type>& right);
	{ return left -= right; }

	template <typename base_type>
	__host__ __device__ auto
	operator*(complex<base_type> left, const complex<base_type>& right)
	{ return left *= right; }

	template <typename base_type>
	__host__ __device__ auto
	operator/(complex<base_type> left, const complex<base_type>& right)
	{ return left /= right; }

	template <typename base_type>
	std::ostream& operator<<(std::ostream& out, const complex<base_type>& z)
	{
		auto re = real(z);
		auto im = imag(z);

		if (re) {
			out << re;
			if (im) out << '+' << im << 'i';
		}
		else if (im) out << im << 'i';
		else out << 0;

		return out;
	}

	template <> struct library_type<complex<double>> :
		std::integral_constant<data_type, data_type::double_precision_complex> {};
	template <> struct library_type<complex<float>> :
		std::integral_constant<data_type, data_type::single_precision_complex> {};

	template <> struct is_numerical_type<complex<double>> : std::true_type {};
	template <> struct is_numerical_type<complex<float>> : std::true_type {};
}
