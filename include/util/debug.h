#pragma once

#include <typeinfo>
#include <cxxabi.h>
#include <string>

namespace dbg {

// Use the c++ ABI library to get typename in human readable format. E.g.,
//
//     std::cout << demangled<T> << '\n'
//

template <typename T>
const char*&
demangle()
{
	static int status;
	static const char* name = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
	return name;
}

template <typename T>
struct demangler {
	static const char* name;

	operator const char*() const { return name; }
	operator std::string() const { return name; }
};

template <typename T>
const char* demangler<T>::name = demangle<T>();

template <typename T>
inline constexpr demangler<T> demangled = {};

}
