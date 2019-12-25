#pragma once
#include "algo/preconditioner.h"
#include "algo/pcg.h"
#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "algo/symmilu.h"
#include "algo/redblack.h"
#include "mg/solver.h"
#include "types.h"

namespace ins {
namespace solvers {

struct solver { virtual vector operator()(vector) const = 0; };

inline auto solve(const solver& slv, vector b) { return slv(std::move(b)); }

namespace __1 {

struct chebyshev : algo::chebyshev, algo::preconditioner {
private:
	chebyshev(std::pair<double, double> range, const algo::matrix& m) :
		algo::chebyshev(range.second, range.first, m) {}
public:
	virtual vector
	operator()(vector b) const
	{
		return algo::chebyshev::operator()(std::move(b));
	}

	chebyshev(const algo::matrix& m) :
		chebyshev(algo::gershgorin(m), m) {}
};

struct ilu : algo::symmilu {
	template <typename grid_type>
	ilu(const grid_type& grid, const algo::matrix& m) :
		algo::symmilu{m, new algo::redblack{grid}} {}
};

struct multigrid : algo::preconditioner, mg::solver {
	virtual vector
	operator()(vector v) const
	{
		return mg::solver::operator()(std::move(v), 1);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	multigrid(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		mg::solver(grid, tolerance, sm_gen, op_gen) {}
};

} // namespace __1


struct chebpcg : solver {
private:
	using chebyshev = __1::chebyshev;
	double tolerance;
	matrix m;
	chebyshev preconditioner;
public:
	virtual vector
	operator()(vector v) const
	{
		using algo::krylov::pcg;
		return pcg(preconditioner, m, std::move(v), tolerance);
	}

	chebpcg(double tolerance, matrix op) :
		tolerance(tolerance),
		m(std::move(op)),
		preconditioner(m) {}
	chebpcg(chebpcg&& o) :
		tolerance(o.tolerance),
		m(std::move(o.m)),
		preconditioner(m) {}
};

struct ilupcg : solver {
private:
	using ilu = __1::ilu;
	double tolerance;
	matrix m;
	ilu preconditioner;
public:
	virtual vector
	operator()(vector v) const
	{
		using algo::krylov::pcg;
		return pcg(preconditioner, m, std::move(v), tolerance);
	}

	template <typename grid_type>
	ilupcg(const grid_type& grid, double tolerance, matrix op) :
		tolerance(tolerance),
		m(std::move(op)),
		preconditioner(grid, m) {}
};

struct mgpcg : solver {
private:
	using multigrid = __1::multigrid;
	double tolerance;
	multigrid preconditioner;
public:
	virtual vector
	operator()(vector v) const
	{
		using algo::krylov::pcg;
		const matrix& m = preconditioner.op();
		return pcg(preconditioner, m, std::move(v), tolerance);
	}

	template <typename grid_type, typename sm_gen_type, typename op_gen_type>
	mgpcg(const grid_type& grid, double tolerance,
			const sm_gen_type& sm_gen, const op_gen_type& op_gen) :
		tolerance(tolerance),
		preconditioner(grid, tolerance, sm_gen, op_gen) {}
};

} // namespace solvers
} // namespace ins
