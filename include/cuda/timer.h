#pragma once
#include <string>
#include "util/log.h"
#include "event.h"
#include "units.h"

namespace cuda {

struct timer {
	std::string id;
	cuda::event start, stop;

	timer(std::string id) :
		id(id) { start.record(); }
	~timer() {
		stop.record();
		util::logging::info(id, ": ", (stop-start) * 1_ms);
	}
};

} // namespace cuda
