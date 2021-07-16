#pragma once
#include "solvers/types.h"

namespace solvers {

struct preconditioner {
	virtual dense::vector operator()(dense::vector) const = 0;
	virtual ~preconditioner() {}
};

inline auto
solve(const preconditioner& pr, dense::vector v)
{
	return pr(std::move(v));
}

namespace preconditioners {

struct identity : preconditioner {
	virtual dense::vector operator()(dense::vector v) const { return v; }
};

} // namespace preconditioners
} // namespace solvers
