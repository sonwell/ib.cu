#pragma once
#include <iostream>
#include "cuda/event.h"
#include "util/sequences.h"
#include "util/functional.h"
#include "util/debug.h"
#include "algo/lu.h"
#include "types.h"
#include "differentiation.h"
#include "fill.h"
#include "rbf.h"
#include "polynomials.h"

namespace bases {
namespace impl {

struct bad_dimension : std::logic_error {
	using std::logic_error::logic_error;
};

struct slice : linalg::base {
private:
	static constexpr auto nt = linalg::operation::non_transpose;

	matrix m;
	int r, c;
	linalg::operation op;

	auto ld() const { return m.rows(); }
	auto* values() const { return m.values() + c * ld() + r; }

	void
	swap(slice&& o)
	{
		std::swap(m, o.m);
		std::swap(r, o.r);
		std::swap(c, o.c);
		std::swap(op, o.op);
		linalg::base::operator=(std::move(o));
	}

	void
	copy(const slice& o)
	{
		m = o.m;
		r = o.r;
		c = o.c;
		op = o.op;
		linalg::base::operator=(o);
	}
public:
	slice& operator=(slice&& o) { swap(std::move(o)); return *this; }
	slice& operator=(const slice& o) { copy(o); return *this; }
	operator matrix() const
	{
		auto n = rows(), m = cols(), lead = ld();
		matrix r{n, m};

		auto* src = values();
		auto* dst = r.values();
		auto k = [=, op=op] __device__ (int tid)
		{
			auto r = tid % n;
			auto c = tid / n;
			if (op != nt) {
				auto t = c;
				c = r;
				r = t;
			}
			dst[tid] = src[r + lead * c];
		};
		util::transform<128, 7>(k, n * m);
		return r;
	}

	slice() : base(0, 0), m{}, r(0), c(0), op(nt) {}
	slice(slice&& o) : base(0, 0) { swap(std::move(o)); }
	slice(const slice& o) : base(0, 0) { copy(o); }
	slice(matrix m, int r, int c, int rows, int cols, linalg::operation op) :
		linalg::base(op == nt ? rows : cols, op == nt ? cols : rows),
		m(std::move(m)), r(r), c(c), op(op)
	{
		if (r + rows > this->m.rows() || c + cols > this->m.cols())
			throw bad_dimension("could not take matrix slice");
	}
	slice(matrix m, int rows, int cols, linalg::operation op=nt) :
		slice(std::move(m), 0, 0, rows, cols, op) {}
	slice(matrix m, linalg::operation op=nt) :
		slice(std::move(m), 0, 0, m.rows(), m.cols(), op) {}

	friend inline matrix operator*(const slice&, const matrix&);
	friend inline vector operator*(const slice&, const vector&);
};

inline matrix
operator*(const slice& sl, const matrix& m)
{
	using linalg::size;
	(void) (size(sl) * size(m));
	cublas::handle h;
	static constexpr double alpha = 1.0;
	static constexpr double beta = 0.0;
	matrix r{sl.rows(), m.cols()};
	cublas::gemm(h, sl.op, cublas::operation::non_transpose,
			sl.rows(), m.cols(), sl.cols(),
			&alpha, sl.values(), sl.ld(), m.values(), m.rows(),
			&beta, r.values(), r.rows());
	return r;
}

vector
operator*(const slice& sl, const vector& x)
{
	using linalg::size;
	(void) (size(sl) * size(x));
	cublas::handle h;
	static constexpr double alpha = 1.0;
	static constexpr double beta = 0.0;
	vector r{sl.rows()};
	cublas::gemv(h, sl.op, sl.rows(), sl.cols(),
			&alpha, sl.values(), sl.ld(), x.values(), 1,
			&beta, r.values(), 1);
	return r;
}

} // namespace impl

// Constructs geometric operators (eval, differentiate, etc.) used in `geometry`
// (geometry.h), except for surface patch areas, which are handled on a
// surface-by-surface basis.
template <int dims>
class operators {
public:
	static constexpr auto dimensions = dims;
private:
	using slice = impl::slice;
	using sequence = util::make_sequence<int, dims>;
	static constexpr sequence seq;

	static constexpr auto nfd = dims;
	static constexpr auto nsd = dims * (dims+1) / 2;
	using fdtype = std::array<slice, nfd>;
	using sdtype = std::array<slice, nsd>;

	template <typename rbf, typename poly>
	static slice
	compute_evaluator(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu)
	{
		static constexpr auto tr = linalg::operation::transpose;
		if (&xd == &xs) return {};
		auto m = solve(lu, fill<dimensions>(xs, xd, phi, p));
		return {std::move(m), xd.rows(), xs.rows(), tr};
	}

	template <typename rbf, typename poly, int ... ns>
	static slice
	compute_derivative(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu, partials<ns...> d)
	{
		static constexpr auto tr = linalg::operation::transpose;
		auto m = solve(lu, fill<dimensions>(xs, xd, diff(phi, d), diff(p, d)));
		return {std::move(m), xd.rows(), xs.rows(), tr};
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_first_derivatives(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu, util::sequence<int, ns...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p, lu, bases::d<ns>)...};
	}

	template <typename rbf, typename poly, int n, int ... ms>
	static auto
	compute_second_derivative(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu, partials<n> d, util::sequence<int, ms...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p, lu,
				bases::d<n> * bases::d<n + ms>)...};
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_second_derivatives(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu, util::sequence<int, ns...>)
	{
		auto ops = std::tuple_cat(compute_second_derivative(xd, xs, phi, p, lu, bases::d<ns>,
					util::make_sequence<int, dimensions-ns>())...);
		auto k = [] (auto ... args) { return std::array{std::move(args)...}; };
		return apply(k, std::move(ops));
	}

	typedef struct {
		slice evaluator;
		fdtype first;
		sdtype second;
	} ops_type;

	template <typename interp, typename eval, typename poly>
	static ops_type
	compute_operators(const matrix& xd, const matrix& xs,
			interp phi, eval psi, poly p)
	{
		ops_type ops;
		{
			auto lu = algo::lu(fill<dimensions>(xd, phi, p));
			ops.evaluator = compute_evaluator(xd, xs, phi, p, lu);
			ops.first = compute_first_derivatives(xd, xs, phi, p, lu, seq);
			ops.second = compute_second_derivatives(xd, xs, phi, p, lu, seq);
		}

		if (&xs == &xd)
			ops.restrictor = ops.evaluator;
		else {
			auto lu = algo::lu(fill<dimensions>(xs, phi, p));
			ops.restrictor = compute_evaluator(xs, xd, phi, p, lu);
		}

		return ops;
	}

	operators(int nd, int ns, ops_type ops, vector weights) :
		data_sites(nd), sample_sites(ns),
		evaluator(std::move(ops.evaluator)),
		first_derivatives(std::move(ops.first)),
		second_derivatives(std::move(ops.second)),
		weights(std::move(weights)) {}
public:
	int data_sites;
	int sample_sites;
	slice evaluator;
	fdtype first_derivatives;
	sdtype second_derivatives;
	vector weights;

	template <meta::rbf interp, meta::rbf eval, meta::polynomial poly>
	operators(const matrix& xd, const matrix& xs, vector weights, interp phi, eval psi, poly p) :
		operators(xd.rows(), xs.rows(), compute_operators(xd, xs, phi, psi, p), std::move(weights)) {}

	template <meta::rbf rbf, meta::polynomial poly>
	operators(const matrix& xd, const matrix& xs, vector weights, rbf phi, poly p) :
		operators(xd, xs, std::move(weights), phi, phi, p) {}
};

} // namespace bases
