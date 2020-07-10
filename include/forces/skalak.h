#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"
#include "tension.h"

namespace forces {

// Skalak Law. See Skalak, et al. (1973).
struct skalak : tension {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area shear;
	energy_per_area bulk;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using container = std::array<double, 5>;
		double e = shear;
		double kappa = bulk;
		auto w = [=] __device__ (double i1, double i2) -> container
		{
			auto w1 = e * (i1 - 1);
			auto w2 =  (kappa * (i2 - 1) - e);
			auto w11 = e;
			auto w12 = 0.0;
			auto w22 = kappa;
			return {w1, w2, w11, w12, w22};
		};
		return tension::operator()(object, w);
	}

	constexpr skalak(energy_per_area shear, energy_per_area bulk) :
		shear(shear), bulk(bulk) {}
};

} // namespace forces
