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

namespace algo {

struct lu_factorization {
	dense_matrix m;
	util::memory<int> p;
	util::memory<int> info;
};

inline lu_factorization
lu(dense_matrix m)
{
	int buffer_size;
	cusolver::dense::handle h;
	cuda::stream stream;
	cusolver::set_stream(h, stream);

	cusolver::getrf_buffer_size(h, m.rows(), m.cols(),
			m.values(), m.rows(), &buffer_size);
	util::memory<int> info(1);
	util::memory<int> pivots(m.rows());
	util::memory<double> buffer(buffer_size);

	cusolver::getrf(h, m.rows(), m.cols(),
			m.values(), m.rows(), buffer, pivots, info);
	return {std::move(m), std::move(pivots), std::move(info)};
}

template <template <typename> typename container>
container<linalg::dense<double>>
solve(const lu_factorization& lu, container<linalg::dense<double>> m)
{
	cusolver::dense::handle h;
	cuda::stream stream;
	cusolver::set_stream(h, stream);

	cusolver::operation op = cusolver::operation::non_transpose;
	cusolver::getrs(h, op, m.rows(), m.cols(), lu.m.values(),
			lu.m.rows(), lu.p, m.values(), m.rows(), lu.info);
	return m;
}

} // namespace algo
