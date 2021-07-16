#pragma once
#include "preconditioners/ilu.h"
#include "types.h"
#include "base.h"
#include "pcg.h"

namespace solvers {

struct ilupcg : solver {
private:
	sparse::matrix m;
	preconditioners::ilu pr;
public:
	using solver::tolerance;

	virtual dense::vector
	operator()(dense::vector v) const
	{
		return pcg(pr, m, std::move(v), tolerance);
	}

	template <typename grid_type>
	ilupcg(const grid_type& grid, double tolerance, sparse::matrix op) :
		solver{tolerance}, m(std::move(op)), pr(grid, m) {}
};

} // namespace solvers
