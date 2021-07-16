#pragma once
#include "preconditioners/multigrid.h"
#include "types.h"
#include "base.h"
#include "pcg.h"

namespace solvers {

struct mgpcg : solver {
private:
	preconditioners::multigrid pr;
public:
	using solver::tolerance;

	virtual dense::vector
	operator()(dense::vector v) const
	{
		return pcg(pr, pr.op(), std::move(v), tolerance);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	mgpcg(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		solver{tolerance}, pr(grid, tolerance, sm_gen, op_gen) {}
};

} // namespace solvers
