#pragma once
#include "solvers/types.h"

namespace solvers {
namespace mg {

// MG smoother base class
struct smoother {
	virtual dense::vector operator()(dense::vector) const = 0;
	virtual ~smoother() = default;
};

inline auto solve(const smoother& sm, dense::vector b) { return sm(std::move(b)); }

} // namespace mg
} // namespace solvers
