#pragma once

namespace cuda {

enum class diagonal_type { non_unit, unit };

inline std::ostream&
operator<<(std::ostream& out, diagonal_type v)
{
	switch (v) {
		case diagonal_type::non_unit:
			return out << "non-unit";
		case diagonal_type::unit:
			return out << "unit";
	}
	throw util::bad_enum_value();
}

}
