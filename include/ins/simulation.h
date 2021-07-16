#pragma once
#include "units.h"

namespace ins {

struct simulation {
	units::time timestep;
	units::time time_scale;
	units::length length_scale;
	units::diffusivity coefficient;
	double tolerance;
};

struct parameters : simulation {
	units::density density;
	units::viscosity viscosity;

	constexpr parameters(units::time k, units::time time,
			units::length length, units::density rho,
			units::viscosity mu, double tol) :
		simulation{k, time, length, mu / rho, tol},
		density(rho), viscosity(mu) {}
};


} // namespace ins
