#pragma once
#include "preconditioners/chebyshev.h"
#include "types.h"
#include "base.h"
#include "pcg.h"

namespace solvers {

struct chebpcg : solver {
private:
	preconditioners::chebyshev pr;
public:
	using solver::tolerance;

	virtual dense::vector
	operator()(dense::vector v) const
	{
		return pcg(pr, pr.op(), std::move(v), tolerance);
	}

	chebpcg(sparse::matrix op, double tolerance) :
		solver{tolerance}, pr(std::move(op)) {}
};

} // namespace solvers
