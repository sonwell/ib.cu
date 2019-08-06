#pragma once
#include <tuple>
#include <functional>
#include "util/functional.h"
#include "bases/geometry.h"
#include "types.h"

namespace forces {

template <typename ... force_types>
struct combine {
	std::tuple<force_types...> forces;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using bases::current;
		using namespace util::functional;
		const auto& curr = object.geometry(current).sample;
		matrix f{linalg::size(curr.position), linalg::zero};
		auto k = [&] (matrix l, const auto& r) { return l + r(object); };
		return apply(partial(foldl, k, std::move(f)), forces);
	}

	constexpr combine(force_types ... forces) :
		forces{forces...} {};
};

}
