#pragma once
#include "solvers/multigrid.h"
#include "solvers/types.h"
#include "base.h"

namespace solvers {
namespace preconditioners {

struct multigrid : preconditioner, solvers::multigrid {
	virtual dense::vector
	operator()(dense::vector v) const
	{
		return solvers::multigrid::operator()(std::move(v), 1);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	multigrid(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		solvers::multigrid(grid, tolerance, sm_gen, op_gen) {}
};

} // namespace preconditioners
} // namespace solvers
