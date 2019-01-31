#pragma once
#include <utility>
#include <ostream>
#include "preconditioner.h"
#include "coloring.h"
#include "device_ptr.h"
#include "lwps/matrix.h"
#include "lwps/vector.h"
#include "util/launch.h"

namespace algo {
__host__ __device__
double
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

class symmilu : public preconditioner {
private:
	using coloring_ptr = std::unique_ptr<coloring>;
	coloring_ptr colorer;
	mem::device_ptr<int> offsets;
	const lwps::matrix lu;
protected:
	void
	update(lwps::matrix& m, int row, int col,
			mem::device_ptr<double>& diagonals) const
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

	lwps::matrix
	factor(const lwps::matrix& m) const
	{
		auto lu = colorer->permute(m);
		auto rows = lu.rows();
		mem::device_ptr<double> diagonals(rows);

		auto* cdata = colorer->starts();
		auto colors = colorer->colors();
		auto* sdata = lu.starts();
		auto* vdata = lu.values();
		auto* idata = offsets.data();
		auto* ddata = diagonals.data();

		auto start = cdata[0];
		auto end = cdata[1];

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
		return std::move(lu);
	}

	lwps::vector
	solve(lwps::vector& v) const
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
			auto start = sdata[cstart + tid];
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

		return std::move(v);
	}
public:
	virtual lwps::vector
	operator()(const lwps::vector& b) const
	{
		auto&& p = colorer->permute(b);
		auto&& y = solve(p);
		return colorer->unpermute(y);
	}

	symmilu(const lwps::matrix& m, coloring_ptr colorer) :
		colorer(std::move(colorer)), offsets(m.rows()), lu(factor(m)) {}
	symmilu(const lwps::matrix& m, coloring* colorer) :
		symmilu(m, coloring_ptr(colorer)) {}

	friend std::ostream& operator<<(std::ostream&, const symmilu&);
};

inline std::ostream&
operator<<(std::ostream& out, const symmilu& ilu)
{
	return out << ilu.lu;
}
}
