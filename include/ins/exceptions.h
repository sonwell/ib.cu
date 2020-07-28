#pragma once
#include <stdexcept>

namespace ins {

struct no_solution : std::runtime_error {
	using std::runtime_error::runtime_error;
};

struct too_small : std::runtime_error {
	too_small(const std::string& whatarg) :
		std::runtime_error(whatarg) {}
	too_small(const char* whatarg = "requested timestep is too small") :
		std::runtime_error(whatarg) {}
};

}
