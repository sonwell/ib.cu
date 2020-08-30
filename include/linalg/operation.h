#pragma once
#include <ostream>

namespace linalg {

enum class operation {
	non_transpose = 'N',
	transpose = 'T',
	conjugate_transpose = 'H'
};

inline std::ostream&
operator<<(std::ostream& out, operation op)
{
	return out << static_cast<char>(op);
}

}
