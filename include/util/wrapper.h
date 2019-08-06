//#include "thrust/tuple.h"
//#include "thrust/functional.h"
#pragma once
#include <tuple>

namespace util {

template <typename /*tag*/, typename wrapped>
struct wrapper : wrapped {
	constexpr explicit operator wrapped&() { return *this; }
	constexpr explicit operator const wrapped&() const { return *this; }

#define operators(transform) \
	transform(=) \
	transform(+=) \
	transform(-=) \
	transform(*=) \
	transform(/=) \
	transform(%=) \
	transform(^=) \
	transform(&=) \
	transform(|=) \
	transform(>>=) \
	transform(<<=)
#define binary(op) \
	template <typename arg_type, \
	          typename return_type = decltype(std::declval<wrapped>() op std::declval<arg_type>()), \
	          typename = std::enable_if_t<std::is_same_v<wrapped&, return_type>>> \
	constexpr decltype(auto) operator op(arg_type&& arg) \
	{ \
		return static_cast<wrapper&>( \
				static_cast<wrapped&>(*this) op std::forward<arg_type>(arg)); \
	}

	operators(binary);

#undef binary
#undef operators

	template <typename ... arg_types>
	constexpr wrapper(arg_types&& ... args) :
		wrapped{std::forward<arg_types>(args)...} {}
};

#define operators(transform) \
	transform(+) \
	transform(-) \
	transform(*) \
	transform(/) \
	transform(%) \
	transform(^) \
	transform(&) \
	transform(|) \
	transform(>>) \
	transform(<<)
#define binary(op) \
	template <typename tag, typename wrapped, typename arg_type, \
		typename return_type = decltype(std::declval<wrapped>() op std::declval<arg_type>()), \
		typename = std::enable_if_t<std::is_convertible_v<return_type, wrapped>>> \
	constexpr decltype(auto) \
	operator op(wrapper<tag, wrapped> wr, arg_type&& arg) \
	{ \
		wr.operator op##=(std::forward<arg_type>(arg)); \
		return wr; \
	}

operators(binary)

#undef binary
#undef operators

template <typename tag, typename wrapped,
	typename return_type = decltype(~std::declval<wrapped>()),
	typename = std::enable_if_t<std::is_convertible_v<return_type, wrapped>>>
constexpr decltype(auto)
operator ~(const wrapper<tag, wrapped>& wr)
{
	return wrapper<tag, wrapped>{~(wrapped&) wr};
}

} // namespace util


template <typename tag, typename wrapped>
class std::tuple_size<util::wrapper<tag, wrapped>> :
	public std::tuple_size<wrapped> {};

template <std::size_t i, typename tag, typename wrapped>
class std::tuple_element<i, util::wrapper<tag, wrapped>> :
	public std::tuple_element<i, wrapped> {};
