#pragma once
#include <memory>
#include <functional>
#include "fd/size.h"
#include "fd/grid.h"
#include "util/functional.h"
#include "mg/smoother.h"
#include "mg/interpolation.h"
#include "types.h"

namespace solvers {

class multigrid;

namespace __1 {

class base_solver {
protected:
	double tolerance;
	sparse::matrix op;

	virtual dense::vector nested_iteration(dense::vector) const = 0;
	virtual dense::vector vcycle(dense::vector) const = 0;
	virtual dense::vector operator()(dense::vector) const = 0;
	virtual dense::vector operator()(dense::vector, int) const = 0;
public:
	virtual ~base_solver() {}
protected:
	template <typename grid_type, typename op_func>
	base_solver(const grid_type& grid, double tolerance, op_func op) :
		tolerance(tolerance), op(op(grid)) {}

friend class solvers::multigrid;
friend class direct_solver;
friend class iterative_solver;
};

using solver_ptr = std::unique_ptr<base_solver>;
using smoother_ptr = std::unique_ptr<mg::smoother>;

// Direct solver: attempt to reduce errors in all modes
// The linear solve should be pretty small at this point, but the operator of
// interest is singular. So, we just use the same relaxer to improve our guess.
// This seems to do well: with the 4th order Chebyshev smoother, we see
// improvement of a factor of ~2500x per complete V-cycle.
class direct_solver : public base_solver {
private:
	dense::vector relax(const dense::vector& v) const { return solve(*sm, v); }
protected:
	using base_solver::op;
	smoother_ptr sm;

	virtual dense::vector nested_iteration(dense::vector v) const { return relax(v); }
	virtual dense::vector vcycle(dense::vector v) const { return relax(v); }
	virtual dense::vector operator()(dense::vector v, int) const { return relax(v); }
	virtual dense::vector operator()(dense::vector v) const { return relax(v); }

	template <typename grid_type, typename op_func, typename sm_func>
	direct_solver(const grid_type& grid, double tolerance, op_func op, sm_func sm) :
		base_solver(grid, tolerance, op), sm(sm(grid, base_solver::op)) {}
friend class solvers::multigrid;
friend class iterative_solver;
};

// Iterative solver: reduce errors in high frequency modes on a refined grid
class iterative_solver : public base_solver {
private:
	smoother_ptr sm;
	sparse::matrix restriction;
	sparse::matrix interpolation;
	solver_ptr coarse;

	dense::vector smooth(const dense::vector& b) const { return solve(*sm, b); }

	dense::vector
	smooth(const dense::vector& x, dense::vector b) const
	{
		return smooth(residual(x, b)) + x;
	}

	inline dense::vector
	residual(const dense::vector& x, dense::vector b) const
	{
		gemv(-1.0, op, x, 1.0, b);
		return b;
	}

	template <typename pred_type>
	dense::vector
	iterate(dense::vector b, pred_type pred) const
	{
		auto x = nested_iteration(b);
		if (!pred(0)) return x;
		auto r = residual(x, std::move(b));
		int iteration = 0;
		while (abs(r) > tolerance) {
			auto e = vcycle(r);
			axpy(1.0, e, x);
			if (!pred(iteration++))
				break;
			gemv(-1.0, op, e, 1.0, r);
		}
		return x;
	}
protected:
	using base_solver::op;

	virtual dense::vector
	nested_iteration(dense::vector b) const
	{
		auto x = restriction * b;
		x = coarse->nested_iteration(std::move(x));
		x = interpolation * x;
		auto r = residual(x, std::move(b));
		x += vcycle(std::move(r));
		return x;
	}

	virtual dense::vector
	vcycle(dense::vector b) const
	{
		auto x = smooth(b);
		auto r = residual(x, b);
		r = restriction * std::move(r);
		r = coarse->vcycle(std::move(r));
		gemv(1.0, interpolation, std::move(r), 1.0, x);
		return smooth(x, std::move(b));
	}

	virtual dense::vector
	operator()(dense::vector b) const
	{
		return iterate(std::move(b), [] (int) { return true; });
	}

	virtual dense::vector
	operator()(dense::vector b, int its) const
	{
		return iterate(std::move(b), [&] (int it) { return it < its; });
	}

	template <typename grid_type, typename op_func, typename sm_func>
	iterative_solver(const grid_type& grid, op_func op, sm_func sm,
			solver_ptr&& coarse) :
		base_solver(grid, coarse->tolerance, op),
		sm(sm(grid, base_solver::op)),
		restriction(mg::restriction(grid)),
		interpolation(mg::interpolation(grid)),
		coarse(std::move(coarse)) {}
friend class solvers::multigrid;
};

template <typename grid_type>
constexpr bool
refined(const grid_type& grid)
{
	// We can coarsen the grid in one dimension if:
	//  * the dimension has an even number of cells
	//  * the dimension is periodic and has at least two cells
	//  * the dimension is non periodic and has at least four cells
	using namespace util::functional;
	auto k = [] (const auto& comp)
	{
		auto pts = comp.cells();
		auto solid = comp.solid_boundary;
		if (pts & 1) return false;
		if (!solid) return pts > 2;
		return pts > 4;
	};
	auto reduce = partial(foldl, std::logical_and<void>(), true);
	return apply(reduce, map(k, grid.components()));
}

} // namespace __1

class multigrid {
private:
	using solver_ptr = __1::solver_ptr;
	solver_ptr slv;

	template <typename grid_type, typename op_func, typename sm_func>
	static solver_ptr
	construct(const grid_type& grid, double tolerance, op_func op, sm_func sm)
	{
		using __1::iterative_solver;
		using __1::direct_solver;
		using __1::refined;
		if (refined(grid)) {
			auto refinement = grid.refinement() >> 1;
			auto coarse = fd::grid{grid, refinement};
			auto solver = construct(coarse, tolerance, op, sm);
			return solver_ptr(new iterative_solver(grid, op, sm, std::move(solver)));
		}
		return solver_ptr(new direct_solver(grid, tolerance, op, sm));
	}

	multigrid&
	swap(multigrid& o)
	{
		std::swap(slv, o.slv);
		return *this;
	}
public:
	dense::vector operator()(dense::vector x) const { return (*slv)(std::move(x)); }
	dense::vector operator()(dense::vector x, int it) const { return (*slv)(std::move(x), it); }
	const sparse::matrix& op() const { return slv->op; }

	template <typename grid_type, typename op_func, typename sm_func>
	multigrid(const grid_type& grid, double tolerance, op_func op, sm_func sm) :
		slv(construct(grid, tolerance, op, sm)) {}
	multigrid(multigrid&& o) : slv(nullptr) { swap(o); }
};

} // namespace solvers
