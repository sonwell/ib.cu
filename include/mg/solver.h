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
namespace __1 {

class solver;

class base_solver {
protected:
	matrix op;

	virtual vector nested_iteration(const vector&) const = 0;
	virtual vector vcycle(const vector&) const = 0;
	virtual vector operator()(const vector&) const = 0;
	virtual vector operator()(const vector&, int) const = 0;
public:
	virtual ~base_solver() {}
protected:
	template <typename grid_type, typename op_func>
	base_solver(const grid_type& grid, op_func op) :
		op(op(grid)) {}

friend class solver;
friend class direct_solver;
friend class iterative_solver;
};

using solver_ptr = std::unique_ptr<base_solver>;
using smoother_ptr = std::unique_ptr<smoother>;

class direct_solver : public base_solver {
protected:
	using base_solver::op;
	smoother_ptr sm;

	virtual vector nested_iteration(const vector& v) const { return operator()(v, 0); }
	virtual vector vcycle(const vector& v) const { return operator()(v, 0); }
	virtual vector operator()(const vector& v, int) const { return solve(*sm, v); }
	virtual vector operator()(const vector& v) const { return operator()(v, 0); }

	template <typename grid_type, typename op_func, typename sm_func>
	direct_solver(const grid_type& grid, op_func op, sm_func sm) :
		base_solver(grid, op), sm(sm(grid, base_solver::op)) {}
friend class solver;
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
	smooth(const vector& x, const vector& b) const
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
	iterate(const vector& b, pred_type pred) const
	{
		int iteration = 0;
		auto fine = nested_iteration(b);
		auto r = residual(fine, b);
		// short-circuit to avoid calling abs
		while (pred(iteration++) && abs(r) > 1e-8) {
			fine += vcycle(r);
			r = residual(fine, b);
		}
		return fine;
	}
protected:
	using base_solver::op;

	vector
	init(const vector& b) const
	{
		auto restricted = restriction * b;
		auto iterated = coarse->nested_iteration(restricted);
		auto interpolated = interpolation * iterated;
		return interpolated;
	}

	virtual vector
	nested_iteration(const vector& b) const
	{
		auto fine = init(b);
		auto r = residual(fine, b);
		fine += vcycle(r);
		return fine;
	}

	virtual vector
	vcycle(const vector& b) const
	{
		auto fine = smooth(b);
		auto r = residual(fine, b);
		auto restricted = restriction * r;
		auto approx = coarse->vcycle(restricted);
		gemv(1.0, interpolation, approx, 1.0, fine);
		return smooth(fine, b);
	}

	virtual vector
	operator()(const vector& b) const
	{
		return iterate(b, [] (int) { return true; });
	}

	virtual vector
	operator()(const vector& b, int its) const
	{
		return iterate(b, [&] (int it) { return it < its; });
	}

	template <typename grid_type, typename op_func, typename sm_func>
	iterative_solver(const grid_type& grid, op_func op, sm_func sm,
			solver_ptr&& coarse) :
		base_solver(grid, op), sm(sm(grid, base_solver::op)),
		restriction(mg::restriction(grid)),
		interpolation(mg::interpolation(grid)),
		coarse(std::move(coarse)) {}
	friend class solver;
};

template <typename grid_type>
constexpr bool
refined(const grid_type& grid)
{
	using namespace util::functional;
	auto k = [] (unsigned pts) { return !(pts & 1) && pts > 4; };
	auto reduce = partial(foldl, std::logical_and<void>(), true);
	return apply(reduce, map(k, fd::__1::sizes(grid)));
}

class solver {
private:
	solver_ptr slv;

	template <typename grid_type, typename op_func, typename sm_func>
	static solver_ptr
	construct(const grid_type& grid, op_func op, sm_func sm)
	{
		if (refined(grid)) {
			auto refinement = grid.refinement() >> 1;
			auto coarse = fd::grid{grid, refinement};
			auto solver = construct(coarse, op, sm);
			return solver_ptr(new iterative_solver(grid, op, sm, std::move(solver)));
		}
		return solver_ptr(new direct_solver(grid, op, sm));
	}

	solver&
	swap(solver& o)
	{
		std::swap(slv, o.slv);
		return *this;
	}
public:
	vector operator()(const vector& x) const { return (*slv)(x); }
	vector operator()(const vector& x, std::size_t it) const { return (*slv)(x, it); }
	const matrix& op() const { return slv->op; }

	template <typename grid_type, typename op_func, typename sm_func>
	solver(const grid_type& grid, op_func op, sm_func sm) :
		slv(construct(grid, op, sm)) {}
	solver(solver&& o) : slv(nullptr) { swap(o); }
};

} // namespace __1

using __1::solver;

} // namespace mg
