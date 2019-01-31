#include <cuComplex.h>
#include <type_traits>


namespace cusparse {
	template <typename> struct is_value_type : std::false_type {};
	template <> struct is_value_type<float> : std::true_type {}
	template <> struct is_value_type<double> : std::true_type {}
	template <> struct is_value_type<cuFloatComplex> : std::true_type {}
	template <> struct is_value_type<cuDoubleComplex> : std::true_type {}

	template <typename type>
	inline constexpr bool is_value_type_v = is_value_type<type>::value;
}
