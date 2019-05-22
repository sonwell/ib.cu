#pragma once
#include "util/array.h"

namespace algo
{

template <typename left_type, typename right_type>
constexpr __host__ __device__ auto
dot(const left_type& left, const right_type& right)
{
	static_assert(sizeof(left) / sizeof(left[0]) == sizeof(right) / sizeof(right[0]));
	constexpr auto n = sizeof(left) / sizeof(left[0]);
	using result_type = decltype(left[0] * right[0]);
	result_type result = 0;
	for (int i = 0; i < n; ++i)
		result += left[i] * right[i];
	return result;
}

}
