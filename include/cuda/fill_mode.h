#pragma once
#include <ostream>

namespace cuda {

enum class fill_mode { lower, upper };

inline std::ostream&
operator<<(std::ostream& out, fill_mode v)
{
	switch (v) {
		case fill_mode::lower:
			return out << "lower";
		case fill_mode::upper:
			return out << "upper";
	}
	throw util::bad_enum_value();
}

}
