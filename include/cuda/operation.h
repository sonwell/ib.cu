#pragma once
#include <ostream>

namespace cuda {

enum class operation {
	non_transpose,
	transpose,
	conjugate_transpose
};

inline
std::ostream&
operator<<(std::ostream& out, operation op)
{
	switch (op) {
		case operation::non_transpose: return out << 'N';
		case operation::transpose: return out << 'T';
		case operation::conjugate_transpose: return out << 'H';
	}
	return out;
}

}
