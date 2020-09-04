#pragma once
#include <type_traits>

namespace bases {

// traits pattern for computing shapes, sampling parameter sites, and computing
// integration weights methods for different surfaces.
template <typename shape_type>
struct traits {
	static constexpr auto dimensions = shape_type::dimensions;

	template <typename ... arg_types>
	static constexpr auto
	sample(const arg_types& ... args)
	{
		return shape_type::sample(args...);
	}

	template <typename ... arg_types>
	static constexpr auto
	shape(const arg_types& ... args)
	{
		return shape_type::shape(args...);
	}

	template <typename ... arg_types>
	static constexpr auto
	weights(const arg_types& ... args)
	{
		return shape_type::weights(args...);
	}
};

template <typename> struct is_traits : std::false_type {};
template <typename T> struct is_traits<traits<T>> : std::true_type {};
template <typename T> inline constexpr bool is_traits_v = is_traits<T>::value;

namespace meta { template <typename T> concept traits = is_traits_v<T>; }

}
