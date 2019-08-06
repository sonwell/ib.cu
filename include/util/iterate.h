#pragma once
#include <type_traits>
#include <array>
#include <utility>

namespace util {
namespace detail {

template <std::size_t n> using iteration =
	std::integral_constant<std::size_t, n>;

template <std::size_t m, typename func, std::size_t ... n>
constexpr auto
iterate(func f, std::index_sequence<n...>) ->
	std::enable_if_t<
		!std::is_void_v<decltype(f(iteration<m>()))>,
		std::array<decltype(f(iteration<m>())), sizeof...(n)>
	>
{
	return {f((iteration<m + n>())...)};
}

template <int m, typename func, int ... n>
constexpr auto
iterate(func f, std::index_sequence<n...> seq) ->
	std::enable_if_t<std::is_void_v<decltype(f(iteration<m>()))>>
{
	iteration<m>([=](auto&& v) { f(std::forward<decltype(v)>(v)); return 0; }, seq);
}

} // namespace detail

template <std::size_t m, std::size_t n, typename func>
constexpr decltype(auto)
iterate(func f)
{
	return detail::iterate<m>(f, std::make_index_sequence<n-m>());
}

template <std::size_t n, typename func>
constexpr decltype(auto)
iterate(func f)
{
	return detail::iterate<0>(f, std::make_index_sequence<n>());
}

} // namespace util
