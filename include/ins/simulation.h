#pragma once
#include "units.h"

namespace ins {

struct simulation {
	units::time timestep;
	units::diffusivity coefficient;
	double tolerance;
};

} // namespace ins
