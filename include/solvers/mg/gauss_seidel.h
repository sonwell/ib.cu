#pragma once
#include <cmath>
#include <limits>
#include "algo/coloring.h"
#include "algo/gershgorin.h"
#include "linalg/size.h"
#include "types.h"
#include "smoother.h"

namespace solvers {
namespace mg {

// Weighted Gauss-Seidel red-black smoother for Laplacian/Helmholtz operators
//
// Suppose x satisfies Ax⃗ = b⃗. Let x⃗₁ be a guess for x⃗. We wish to construct an
// improved guess, x⃗₂. We do this by splitting A:
//
//     Ax⃗ = [(1/ω)(D-L) + (1-1/ω)(D-L) - U]x⃗ = b⃗,
//
// where A = D-L-U and D is diagonal, L is strictly lower-triangular, and U is
// strictly upper-triangular. We construct an iteration scheme:
//
//     (1/ω)(D-L)x⃗₂ = b⃗  - [(1-1/ω)(D-L) - U]x⃗₁
//                  = b⃗  - Ax⃗₁ + (1/ω)(D-L)x⃗₁
//         =>    x⃗₂ = x⃗₁ + ω(D-L)⁻¹(b⃗ - Ax⃗₁).
//
// The algorithm proceeds blockwise. Since the red-black coloring makes the
// diagonal blocks diagonal matrices, ω(D-L)⁻¹ ends up being very simple.
class gauss_seidel : public smoother {
private:
	using coloring_ptr = std::unique_ptr<algo::coloring>;

	coloring_ptr colorer;
	sparse::matrix op;
	double omega;

	static double
	compute_omega(const sparse::matrix& m)
	{
		// A guess at the weighting parameter.
		auto [lower, upper] = algo::gershgorin(m);
		auto diag = (lower + upper) / 2;
		auto u = abs(lower) > abs(upper) ? lower : upper;
		auto rho = u / diag - 1;
		auto mu = rho / (1 + sqrt(1 - rho * rho));
		auto eps = std::numeric_limits<double>::epsilon();
		return (1 + mu * mu) * (1 - eps);
	}

	void
	iterate_block_row(int color, const dense::vector& b, dense::vector& x) const
	{
		auto* cstarts = colorer->starts();
		auto* starts = op.starts();
		auto* indices = op.indices();
		auto* values = op.values();
		auto* bvalues = b.values();
		auto* xvalues = x.values();

		auto cstart = cstarts[color];
		auto cend = cstarts[color+1];
		auto count = cend - cstart;

		auto k = [=, omega=omega] __device__ (int tid)
		{
			auto row = cstart + tid;
			auto start = starts[row];
			auto end = starts[row+1];
			auto orig = bvalues[row];
			auto value = orig;
			for (auto i = start; i < end; ++i) {
				auto col = indices[i];
				auto val = values[i];
				if (col == row) {
					value /= (omega * val);
					break;
				}
				value -= val * xvalues[col];
			}
			//auto diag = omega * values[i];
			//xvalues[row] = value / diag;
			xvalues[row] = value;
		};
		util::transform<128, 7>(k, count);
	}

	dense::vector
	iterate(const dense::vector& r) const
	{
		dense::vector e{linalg::size(r)};
		auto colors = colorer->colors();
		for (int i = 0; i < colors; ++i)
			iterate_block_row(i, r, e);
		return e;
	}
public:
	virtual dense::vector
	operator()(dense::vector b) const
	{
		auto r = colorer->permute(std::move(b));
		auto x = iterate(std::move(r));
		return colorer->unpermute(std::move(x));
	}

	gauss_seidel(const sparse::matrix& m, double omega, coloring_ptr co) :
		colorer(std::move(co)), op(colorer->permute(m)), omega(omega) {}

	gauss_seidel(const sparse::matrix& m, coloring_ptr co):
		gauss_seidel(m, compute_omega(m), std::move(co)) {}

	gauss_seidel(const sparse::matrix& m, algo::coloring* colorer) :
		gauss_seidel(m, coloring_ptr(colorer)) {}

	gauss_seidel(const sparse::matrix& m, double omega, algo::coloring* colorer) :
		gauss_seidel(m, omega, coloring_ptr(colorer)) {}
};

} // namespace mg
} // namespace solvers
