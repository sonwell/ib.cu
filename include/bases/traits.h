#pragma once

namespace bases {

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

}
