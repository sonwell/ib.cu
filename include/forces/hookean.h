#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"
#include "tension.h"

namespace forces {

// This isn't quite right...
struct hookean : tension {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area shear;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using container = std::array<double, 5>;
		double e = shear;
		auto w = [=] __device__ (double i1, double) -> container
		{
			auto j = sqrt(i1);
			auto w1 = e * (1 - 1. / j);
			auto w2 = 0.0;
			auto w11 = e / (2 * i1 * j);
			auto w12 = 0.0;
			auto w22 = 0.0;
			return {w1, w2, w11, w12, w22};
		};
		return tension::operator()(object, w);
	}

	constexpr hookean(energy_per_area shear) :
		shear(shear) {}
};

struct hookean1d : tension1d {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area shear;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using container = std::array<double, 2>;
		auto e = shear;
		auto w = [=] __device__ (double i1) -> container
		{
			auto j = sqrt(i1);
			auto w1 = e * (1 - 1. / j);
			auto w11 = e / (2 * i1 * j);
			return {w1, w11};
		};
		return tension1d::operator()(object, w);
	}

	constexpr hookean1d(energy_per_area shear) :
		shear(shear) {}
};

} // namespace forces
