#pragma once
#include <utility>
#include "types.h"
#include "util/memory.h"
#include "cuda/stream.h"
#include "cusolver/handle.h"
#include "cusolver/stream.h"
#include "cusolver/operation.h"
#include "cusolver/getrf.h"
#include "cusolver/getrs.h"

namespace solvers {

struct lu {
	util::memory<int> pivots;
	util::memory<int> info;
	dense::matrix m;

	lu(dense::matrix op) :
		pivots(op.rows()), info(1), m(std::move(op))
	{
		int buffer_size;
		cusolver::dense::handle h;
		cuda::stream stream;
		cusolver::set_stream(h, stream);

		cusolver::getrf_buffer_size(h, m.rows(), m.cols(),
				m.values(), m.rows(), &buffer_size);
		util::memory<double> buffer(buffer_size);

		cusolver::getrf(h, m.rows(), m.cols(),
				m.values(), m.rows(), buffer, pivots, info);
	}
};

template <template <typename> typename container>
container<linalg::dense<double>>
solve(const lu& lu, container<linalg::dense<double>> m)
{
	cusolver::dense::handle h;
	cuda::stream stream;
	cusolver::set_stream(h, stream);

	cusolver::operation op = cusolver::operation::non_transpose;
	cusolver::getrs(h, op, m.rows(), m.cols(), lu.m.values(),
			lu.m.rows(), lu.pivots, m.values(), m.rows(), lu.info);
	return m;
}

} // namespace solvers
