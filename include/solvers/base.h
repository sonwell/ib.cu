#pragma once
#include "types.h"

namespace solvers {

struct solver {
	double tolerance;

	virtual dense::vector operator()(dense::vector b) const = 0;
	constexpr solver(double t) : tolerance(t) {}
};

inline auto
solve(const solver& slv, dense::vector b)
{
	return slv(std::move(b));
}

} // namespace solvers
