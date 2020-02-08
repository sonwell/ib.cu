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
	operator()(const object_type& obj) const
	{
		if constexpr (!sizeof...(force_types)) {
			using bases::current;
			const auto& curr = obj.geometry(current).sample;
			return matrix{linalg::size(curr.position), linalg::zero};
		} else {
			using namespace util::functional;
			auto k = [&] (matrix l, auto& r) { return std::move(l) + r(obj); };
			auto m = [&] (auto& f, auto& ... r) { return foldl(k, f(obj), r...); };
			return apply(m, forces);
		}
	}

	constexpr combine(force_types ... forces) :
		forces{forces...} {};
};

}
