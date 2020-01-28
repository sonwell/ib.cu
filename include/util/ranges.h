#pragma once
#include <iostream>
#include "debug.h"
#include "iterators.h"

namespace util {
namespace ranges {

template <typename transform_type>
struct transform {
	transform_type fn;
	constexpr transform(transform_type transform) :
		fn(std::move(transform)) {}
};

template <typename iterable_type, typename transform_type>
constexpr auto
operator|(iterable_type iterable, transform<transform_type> tr)
{
	return iterators::transform{std::move(iterable), tr.fn};
}

inline constexpr struct {} enumerate;

template <typename iterable_type>
constexpr auto
operator|(iterable_type iterable, const decltype(enumerate)&)
{
	return iterators::enumerate{std::move(iterable)};
}

}
} // namespace util
