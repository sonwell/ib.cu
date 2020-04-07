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

struct solver {
	double tolerance;

	virtual vector operator()(vector) const = 0;
	constexpr solver(double t) : tolerance(t) {}
};

inline auto solve(const solver& slv, vector b) { return slv(std::move(b)); }

namespace __1 {

struct chebyshev : algo::chebyshev, algo::preconditioner {
private:
	chebyshev(std::pair<double, double> range, algo::matrix m) :
		algo::chebyshev(range.second, range.first, std::move(m)) {}
protected:
	using algo::chebyshev::m;
public:
	const matrix& op() const { return m; }

	virtual vector
	operator()(vector b) const
	{
		return algo::chebyshev::operator()(std::move(b));
	}

	chebyshev(algo::matrix m) :
		chebyshev(algo::gershgorin(m), std::move(m)) {}
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
	chebyshev pr;
public:
	using solver::tolerance;

	virtual vector
	operator()(vector v) const
	{
		using algo::krylov::pcg;
		return pcg(pr, pr.op(), std::move(v), tolerance);
	}

	chebpcg(double tolerance, matrix op) :
		solver{tolerance},
		pr(std::move(op)) {}
};

struct ilupcg : solver {
private:
	using ilu = __1::ilu;
	matrix m;
	ilu preconditioner;
public:
	using solver::tolerance;

	virtual vector
	operator()(vector v) const
	{
		using algo::krylov::pcg;
		return pcg(preconditioner, m, std::move(v), tolerance);
	}

	template <typename grid_type>
	ilupcg(const grid_type& grid, double tolerance, matrix op) :
		solver{tolerance},
		m(std::move(op)),
		preconditioner(grid, m) {}
};

struct mgpcg : solver {
private:
	using multigrid = __1::multigrid;
	multigrid preconditioner;
public:
	using solver::tolerance;

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
		solver{tolerance},
		preconditioner(grid, tolerance, sm_gen, op_gen) {}
};

} // namespace solvers
} // namespace ins
