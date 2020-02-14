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

} // namespace ins
