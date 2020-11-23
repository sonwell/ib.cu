#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct tether {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area modulus;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object, const matrix&) const
	{
		using bases::reference;
		using bases::current;

		const auto& orig = object.geometry(reference).sample;
		const auto& curr = object.geometry(current).sample;
		return (double) modulus * (orig.position - curr.position);
	}

	constexpr tether(energy_per_area modulus) :
		modulus(modulus) {}
};

} // namespace force
