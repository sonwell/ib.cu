#pragma once
#include <stdexcept>
#include <string>

namespace linalg {

struct not_implemented : std::runtime_error {
	explicit not_implemented(const char* what_arg):
		std::runtime_error(what_arg) {}
	explicit not_implemented(const std::string& what_arg):
		std::runtime_error(what_arg) {}
};

struct mismatch : std::runtime_error {
	explicit mismatch(const char* what_arg):
		std::runtime_error(what_arg) {}
	explicit mismatch(const std::string& what_arg):
		std::runtime_error(what_arg) {}
};

} // namespace linalg
