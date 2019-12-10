#pragma once
#include "algo/preconditioner.h"
#include "algo/pcg.h"
#include "mg/solver.h"
#include "types.h"

namespace ins {
namespace __1 {

struct multigrid_preconditioner : algo::preconditioner, mg::solver {
	virtual vector
	operator()(const vector& v) const
	{
		return mg::solver::operator()(v, 1);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	multigrid_preconditioner(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		mg::solver(grid, tolerance, sm_gen, op_gen) {}
};

struct mgpcg {
private:
	double tolerance;
	multigrid_preconditioner preconditioner;
public:
	vector
	operator()(const vector& v) const
	{
		using algo::krylov::pcg;
		const matrix& m = preconditioner.op();
		return pcg(preconditioner, m, v, tolerance);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	mgpcg(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		tolerance(tolerance),
		preconditioner(grid, tolerance, sm_gen, op_gen) {}
};

inline decltype(auto)
solve(const mgpcg& solver, const vector& b)
{
	return solver(b);
}

} // namespace __1

using __1::mgpcg;

} // namespace ins
