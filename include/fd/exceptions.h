#pragma once
#include <stdexcept>

namespace fd {

struct no_such_dimension : public std::runtime_error {
	no_such_dimension(const char* what_arg) :
		std::runtime_error(what_arg) {}
	no_such_dimension(const std::string& what_arg) :
		std::runtime_error(what_arg) {}
	no_such_dimension() :
		no_such_dimension("No such dimension") {}
};

struct bad_grid_points : std::runtime_error {
	bad_grid_points(const char* what_arg) :
		std::runtime_error(what_arg) {}
};

}
