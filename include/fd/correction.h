#pragma once
#include <cstddef>

namespace fd {
namespace correction {

template <std::size_t n> struct order :
	std::integral_constant<std::size_t, n> {};
inline constexpr order<0> zeroth_order;
inline constexpr order<1> first_order;
inline constexpr order<2> second_order;

} // namespace correction
} // namespace fd
