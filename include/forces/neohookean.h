#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"
#include "tension.h"

namespace forces {

struct neohookean : tension {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area shear;
	energy_per_area bulk;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object, const matrix&) const
	{
		using container = std::array<double, 5>;
		double e = shear;
		double kappa = bulk;
		auto w = [=] __device__ (double i1, double i2) -> container
		{
			auto j = sqrt(i2);
			auto w1 = e / j;
			auto w2 = kappa * (1 - 1/j) - 0.5 * w1 * i1 / i2;
			auto w11 = 0.0;
			auto w12 = -0.5 * w1 / i2;
			auto w22 = 0.75 * w1 * i1 / (i2 * i2) + 0.5 * kappa / (i2 * j);
			return {w1, w2, w11, w12, w22};
		};
		return tension::operator()(object, w);
	}

	constexpr neohookean(energy_per_area shear, energy_per_area bulk) :
		shear(shear), bulk(bulk) {}
};

} // namespace forces
