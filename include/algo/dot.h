#pragma once
#include "util/functional.h"

namespace algo {

// Vector dot product
template <typename left_type, typename right_type>
constexpr decltype(auto)
dot(const left_type& left, const right_type& right)
{
	using namespace util::functional;
	auto k = [] (const auto& l, const auto& r) { return l * r; };
	return apply(partial(foldl, std::plus<void>{}), map(k, left, right));
}

} // namespace algo
