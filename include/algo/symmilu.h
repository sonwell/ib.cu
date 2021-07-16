#pragma once
#include <utility>
#include <ostream>
#include "util/launch.h"
#include "util/memory.h"
#include "types.h"
#include "preconditioner.h"
#include "coloring.h"

namespace algo {

// Sparse-sparse dot product
__host__ __device__ double
spdot(double* vl, int* il, int jl,
		double* vr, int* ir, int jr, double* d)
{
	double value = 0;
	while (*il < jl && *ir < jr) {
		auto ladv = *il <= *ir;
		auto radv = *ir <= *il;
		if (ladv && radv)
			value += (*vr) * (*vl) * d[*il];
		il += ladv;
		vl += ladv;
		ir += radv;
		vr += radv;
	}
	return value;
}

// Computes the ILU(0) of a symmetric sparse matrix by permuting into k-colored
// ordering (see redblack.h). Let PAP^T be the permuted matrix. Its ILU(0) is
//
//           [ B11 B21^T ... Bk1^T ]   [ I1             ] [ U11 U12 ... U1k ]
//   PAP^T = [ B21 B22   ... Bk2^T ] = [ L21 I2         ] [     U22 ... U2k ]
//           [ ... ...   ... ...   ]   [ ... ... ...    ] [         ... ... ]
//           [ Bk1 Bk2   ... Bkk   ]   [ Lk1 Lk2 ... Ik ] [             Ukk ]
//
// In general:
//   Uij = Bji^T - Li1 U1j - Li2 U2j - ... - Li(i-1) U(i-1)j   i <= j
//   Lij = (Bij -  Li1 U1j - Li2 U2j - ... - Li(j-1) U(j-1)j   i > j
//
// Note: (Lik Ukj)^T = Bjk Ukk^{-1} Bik^T = Ljk Uki
// So: (Lij Ujj)^T = Uji, i < j, and we can compute just U and recover L.
//
// The solve is performed block-wise where blocks correspond to colors.

class symmilu {
private:
	using coloring_ptr = std::unique_ptr<coloring>;
	coloring_ptr colorer;
	util::memory<int> offsets;
	matrix lu;
protected:
	void
	update(matrix& m, int row, int col,
			util::memory<double>& diagonals) const
	{
		auto* sdata = m.starts();
		auto* idata = m.indices();
		auto* vdata = m.values();
		auto* cdata = colorer->starts();
		auto* ddata = diagonals.data();
		auto* jdata = offsets.data();

		auto rstart = cdata[row];
		auto rend = cdata[row+1];
		auto cstart = cdata[col];
		auto cend = cdata[col+1];

		auto k = [=] __device__ (int tid)
		{
			auto row = rstart + tid;
			auto start = sdata[row];
			auto end = sdata[row+1];
			auto offset = jdata[row];

			while (offset < end && idata[offset] < cend) {
				auto col = idata[offset];
				auto sdcol = sdata[col];

				auto value = spdot(vdata + start, idata + start, rstart,
						vdata + sdcol, idata + sdcol, cstart, ddata);
				vdata[offset] -= value;
				if (col == row) ddata[row] = vdata[offset];
				if (col < row) vdata[offset] /= ddata[col];
				++offset;
			}
			jdata[row] = offset;
		};
		util::transform(k, rend - rstart);
	}

	matrix
	factor(const matrix& m) const
	{
		auto lu = colorer->permute(m);
		auto rows = lu.rows();
		util::memory<double> diagonals(rows);

		auto colors = colorer->colors();
		auto* sdata = lu.starts();
		auto* vdata = lu.values();
		auto* idata = offsets.data();
		auto* ddata = diagonals.data();

		auto k = [=] __device__ (int tid)
		{
			auto start = sdata[tid];
			idata[tid] = start;
			ddata[tid] = vdata[start];
		};
		util::transform<128, 7>(k, rows);

		for (int col = 0; col < colors; ++col)
			for (int row = 1; row < colors; ++row)
				update(lu, row, col, diagonals);
		return lu;
	}

	vector
	solve(vector v) const
	{
		auto* cdata = colorer->starts();
		auto colors = colorer->colors();
		auto* sdata = lu.starts();
		auto* idata = lu.indices();
		auto* vdata = lu.values();
		auto* wdata = v.values();
		auto* jdata = offsets.data();
		auto rend = cdata[colors];

		auto up = [=] __device__ (int tid, int cend)
		{
			auto start = sdata[cend + tid];
			auto end = sdata[cend + tid+1];
			auto offset = start;
			double value = wdata[cend + tid];

			for (int i = start; i < end && idata[i] < cend; ++i, ++offset)
				value -= vdata[i] * wdata[idata[i]];
			wdata[cend + tid] = value;
			jdata[cend + tid] = offset;
		};

		auto down = [=] __device__ (int tid, int cstart)
		{
			auto offset = jdata[cstart + tid];
			auto end = sdata[cstart + tid+1];
			double value = wdata[cstart + tid];
			double diagonal = vdata[offset];

			for (int i = offset+1; i < end; ++i)
				value -= vdata[i] * wdata[idata[i]];
			wdata[cstart + tid] = value / diagonal;
		};

		for (int col = 0; col < colors; ++col) {
			auto cend = cdata[col+1];
			util::transform(up, rend - cend, cend);
		}

		for (int col = colors-1; col >= 0; --col) {
			auto cstart = cdata[col];
			auto cend = cdata[col+1];
			util::transform(down, cend - cstart, cstart);
		}
		return v;
	}
public:
	vector
	operator()(vector b) const
	{
		auto p = colorer->permute(std::move(b));
		auto y = solve(std::move(p));
		return colorer->unpermute(std::move(y));
	}

	symmilu(const matrix& m, coloring_ptr colorer) :
		colorer(std::move(colorer)), offsets(m.rows()), lu(factor(m)) {}
	symmilu(const matrix& m, coloring* colorer) :
		symmilu(m, coloring_ptr(colorer)) {}
};

} // namespace algo
