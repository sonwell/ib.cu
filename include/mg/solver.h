#pragma once
#include <memory>
#include <functional>
#include "fd/size.h"
#include "fd/grid.h"
#include "util/functional.h"
#include "cuda/event.h"
#include "types.h"
#include "smoother.h"
#include "interpolation.h"

namespace mg {

class solver;

namespace __1 {

class base_solver {
protected:
	double tolerance;
	matrix op;

	virtual vector nested_iteration(vector) const = 0;
	virtual vector vcycle(vector) const = 0;
	virtual vector operator()(vector) const = 0;
	virtual vector operator()(vector, int) const = 0;
public:
	virtual ~base_solver() {}
protected:
	template <typename grid_type, typename op_func>
	base_solver(const grid_type& grid, double tolerance, op_func op) :
		tolerance(tolerance), op(op(grid)) {}

friend class mg::solver;
friend class direct_solver;
friend class iterative_solver;
};

using solver_ptr = std::unique_ptr<base_solver>;
using smoother_ptr = std::unique_ptr<smoother>;

class direct_solver : public base_solver {
private:
	vector relax(const vector& v) const { return solve(*sm, v); }
protected:
	using base_solver::op;
	smoother_ptr sm;

	virtual vector nested_iteration(vector v) const { return relax(v); }
	virtual vector vcycle(vector v) const { return relax(v); }
	virtual vector operator()(vector v, int) const { return relax(v); }
	virtual vector operator()(vector v) const { return relax(v); }

	template <typename grid_type, typename op_func, typename sm_func>
	direct_solver(const grid_type& grid, double tolerance, op_func op, sm_func sm) :
		base_solver(grid, tolerance, op), sm(sm(grid, base_solver::op)) {}
friend class mg::solver;
friend class iterative_solver;
};

class iterative_solver : public base_solver {
private:
	smoother_ptr sm;
	matrix restriction;
	matrix interpolation;
	solver_ptr coarse;

	vector smooth(const vector& b) const { return solve(*sm, b); }

	vector
	smooth(const vector& x, vector b) const
	{
		return smooth(residual(x, b)) + x;
	}

	inline vector
	residual(const vector& x, vector b) const
	{
		gemv(-1.0, op, x, 1.0, b);
		return b;
	}

	template <typename pred_type>
	vector
	iterate(vector b, pred_type pred) const
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

	virtual vector
	nested_iteration(vector b) const
	{
		auto x = restriction * b;
		x = coarse->nested_iteration(std::move(x));
		x = interpolation * x;
		auto r = residual(x, std::move(b));
		x += vcycle(std::move(r));
		return x;
	}

	virtual vector
	vcycle(vector b) const
	{
		auto x = smooth(b);
		auto r = residual(x, b);
		r = restriction * std::move(r);
		r = coarse->vcycle(std::move(r));
		gemv(1.0, interpolation, std::move(r), 1.0, x);
		return smooth(x, std::move(b));
	}

	virtual vector
	operator()(vector b) const
	{
		return iterate(std::move(b), [] (int) { return true; });
	}

	virtual vector
	operator()(vector b, int its) const
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
	friend class mg::solver;
};

template <typename grid_type>
constexpr bool
refined(const grid_type& grid)
{
	using namespace util::functional;
	auto k = [] (const auto& comp)
	{
		auto pts = comp.cells();
		auto solid = comp.solid_boundary;
		if (pts & 1) return false;
		if (solid) return pts > 2;
		return pts > 4;
	};
	auto reduce = partial(foldl, std::logical_and<void>(), true);
	return apply(reduce, map(k, grid.components()));
}

} // namespace __1

class solver {
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

	solver&
	swap(solver& o)
	{
		std::swap(slv, o.slv);
		return *this;
	}
public:
	vector operator()(vector x) const { return (*slv)(std::move(x)); }
	vector operator()(vector x, int it) const { return (*slv)(std::move(x), it); }
	const matrix& op() const { return slv->op; }

	template <typename grid_type, typename op_func, typename sm_func>
	solver(const grid_type& grid, double tolerance, op_func op, sm_func sm) :
		slv(construct(grid, tolerance, op, sm)) {}
	solver(solver&& o) : slv(nullptr) { swap(o); }
};

} // namespace mg
