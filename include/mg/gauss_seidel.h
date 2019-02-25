#pragma once
#include <cmath>
#include <limits>
#include "algo/coloring.h"
#include "algo/gershgorin.h"
#include "smoother.h"

namespace mg {
	class gauss_seidel : public smoother {
	private:
		using coloring_ptr = std::unique_ptr<algo::coloring>;

		coloring_ptr colorer;
		lwps::matrix op;
		double omega;

		static double
		compute_omega(const lwps::matrix& m)
		{
			auto [lower, upper] = algo::gershgorin(m);
			auto diag = (lower + upper) / 2;
			auto u = abs(lower) > abs(upper) ? lower : upper;
			auto rho = u / diag - 1;
			auto mu = rho / (1 + sqrt(1 - rho * rho));
			auto eps = std::numeric_limits<double>::epsilon();
			return (1 + mu * mu) * (1 - eps);
		}

		void
		iterate_block_row(int color, const lwps::vector& b, lwps::vector& x) const
		{
			auto colors = colorer->colors();
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

		lwps::vector
		iterate(const lwps::vector& r) const
		{
			lwps::vector e(size(r));
			auto colors = colorer->colors();
			for (int i = 0; i < colors; ++i)
				iterate_block_row(i, r, e);
			return std::move(e);
		}
	public:
		virtual lwps::vector
		operator()(const lwps::vector& b) const
		{
			auto&& r = colorer->permute(b);
			auto&& x = iterate(r);
			return colorer->unpermute(x);

			// XXX maybe one too many vector allocations?
			/*
			lwps::vector x(size(b), lwps::fill::zeros);
			int iteration = 0;
			while (iterations) {
			for (int it = 0; it < iterations; ++it) {
				if (it) lwps::gemv(-1.0, op, x, 1.0, r);
				x += iterate(r);
			}
			return colorer->unpermute(x);
			*/
		}

		gauss_seidel(const lwps::matrix& m, double omega, coloring_ptr co) :
			colorer(std::move(co)), op(colorer->permute(m)), omega(omega) {}

		gauss_seidel(const lwps::matrix& m, coloring_ptr co):
			gauss_seidel(m, compute_omega(m), std::move(co)) {}

		gauss_seidel(const lwps::matrix& m, algo::coloring* colorer) :
			gauss_seidel(m, coloring_ptr(colorer)) {}

		gauss_seidel(const lwps::matrix& m, double omega, algo::coloring* colorer) :
			gauss_seidel(m, omega, coloring_ptr(colorer)) {}
	};
}
