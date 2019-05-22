#pragma once
#include <stdexcept>

namespace fd {

class no_such_dimension : public std::runtime_error {
	public:
		no_such_dimension() : std::runtime_error("No such dimension.") {}
		no_such_dimension(const char* what_arg) : std::runtime_error(what_arg) {}
		no_such_dimension(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

struct bad_grid_points : std::runtime_error
{
	bad_grid_points(const char* what_arg) :
		std::runtime_error(what_arg) {}
};

}
