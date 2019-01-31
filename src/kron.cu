#include "lwps/matrix.h"
#include "util/launch.h"

namespace lwps {
	matrix
	kron(const matrix& left, const matrix& right)
	{
		enum { nt = 128, vt = 7, nv = nt * vt };

		const auto lrows = left.rows();
		const auto lcols = left.cols();
		const auto lnnz = left.nonzero();
		const auto* lstarts = left.starts();
		const auto* lindices = left.indices();
		const auto* lvalues = left.values();

		const auto rrows = right.rows();
		const auto rcols = right.cols();
		const auto rnnz = right.nonzero();
		const auto* rstarts = right.starts();
		const auto* rindices = right.indices();
		const auto* rvalues = right.values();

		auto rows = lrows * rrows;
		auto cols = lcols * rcols;
		auto nnz = lnnz * rnnz;
		matrix result{rows, cols, nnz};
		auto* starts = result.starts();
		auto* indices = result.indices();
		auto* values = result.values();

		auto r = [=] __device__ (int tid)
		{
			int rrow = tid % rrows;
			int lrow = tid / rrows;
			auto lcurr = lstarts[lrow];
			auto lnext = lstarts[lrow+1];
			auto rcurr = rstarts[rrow];

			auto start = rnnz * lcurr + rcurr * (lnext - lcurr);
			starts[tid] = start;
			if (!tid) starts[rows] = nnz;
		};

		auto v = [=] __device__ (int tid)
		{
			auto f = [&] ()
			{
				auto start = 0;
				auto end = rows+1;

				while (start != end) {
					auto middle = (start + end) >> 1;
					auto el = starts[middle];
					auto pred = el > tid;
					start = pred ? start : middle+1;
					end = pred ? middle : end;
				}
				return start-1;
			};

			int row = f();
			int rrow = row % rrows;
			int lrow = row / rrows;
			auto rcurr = rstarts[rrow];
			auto lcurr = lstarts[lrow];
			auto rnext = rstarts[rrow+1];
			auto rdiff = rnext - rcurr;

			auto start = starts[row];
			int offset = tid - start;
			auto roffset = rcurr + offset % rdiff;
			auto loffset = lcurr + offset / rdiff;

			auto rcol = rindices[roffset];
			auto lcol = lindices[loffset];
			auto rval = rvalues[roffset];
			auto lval = lvalues[loffset];

			auto index = rcol + rcols * lcol;
			auto value = rval * lval;


			indices[tid] = rcol + rcols * lcol;
			values[tid] = rval * lval;
		};

		util::transform<nt, vt>(r, rows);
		util::transform<nt, vt>(v, nnz);
		cudaDeviceSynchronize();
		return std::move(result);
	}
}
