#pragma once
#include <cstddef>
#include "complex.h"
#include "cusparse/index_base.h"

namespace linalg {

static constexpr auto index_base = cusparse::index_base::zero;

template <typename> struct is_field : std::false_type {};
template <> struct is_field<double> : std::true_type {};
template <> struct is_field<float> : std::true_type {};
template <> struct is_field<complex<double>> : std::true_type {};
template <> struct is_field<complex<float>> : std::true_type {};

template <typename type>
inline constexpr auto is_field_v = is_field<type>::value;

template <typename type> struct is_scalar :
	std::integral_constant<bool, std::is_arithmetic_v<type> ||
		is_field_v<type>> {};

template <typename type>
inline constexpr auto is_scalar_v =  is_scalar<type>::value;

template <typename value_type>
struct base_type { using type = value_type; };

template <typename value_type>
struct base_type<complex<value_type>> { using type = value_type; };

template <typename value_type>
using base_t = typename base_type<value_type>::type;

template <typename value_type,
		 typename = std::enable_if_t<is_scalar_v<value_type>>>
struct scalar_type { using type = value_type; };

template <typename value_type>
using scalar = typename scalar_type<value_type>::type;

}
