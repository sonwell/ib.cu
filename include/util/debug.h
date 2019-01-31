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
}
