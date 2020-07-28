#pragma once
#include <stdexcept>

namespace ins {

struct no_solution : std::runtime_error {
	using std::runtime_error::runtime_error;
};

}
