#pragma once

#include <typeinfo>
#include <cxxabi.h>

namespace dbg {

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

	operator const char*() const { return name; };
};

template <typename T>
const char* demangler<T>::name = demangle<T>();

template <typename T>
inline constexpr demangler<T> demangled = {};

}
