#pragma once
#include <utility>
#include "types.h"
#include "util/memory.h"
#include "cuda/stream.h"
#include "cublas/stream.h"
#include "cublas/gemm.h"
#include "cublas/operation.h"
#include "cublas/handle.h"
#include "cusolver/stream.h"
#include "cusolver/handle.h"
#include "cusolver/gesvd.h"

namespace solvers {

struct svd {
	util::memory<int> info;
	dense::matrix u;
	dense::matrix vt;
	dense::vector sigma;

	svd(const dense::matrix& m) :
		info(1), u(m.rows(), m.rows()), vt(m.cols(), m.cols()),
		sigma(std::min(m.rows(), m.cols()))
	{
		int buffer_size;
		cusolver::dense::handle h;
		cuda::stream stream;
		cusolver::set_stream(h, stream);

		cusolver::gesvd_buffer_size(h, m.rows(), m.cols(),
				m.values(), m.rows(), &buffer_size);
		auto k = m.rows() < m.cols() ? m.rows() : m.cols();
		util::memory<double> buffer(buffer_size);
		util::memory<double> rwork(k-1);

		cusolver::gesvd(h, cusolver::job::all, cusolver::job::all, m.rows(), m.cols(),
				m.values(), m.rows(), sigma.values(), u.values(), u.rows(),
				vt.values(), vt.rows(), buffer, buffer_size, rwork, info);
	}
};

template <template <typename> typename container>
container<linalg::dense<double>>
solve(const svd& svd, container<linalg::dense<double>> m)
{
	cublas::handle h;
	cuda::stream stream;
	cublas::set_stream(h, stream);

	const double alpha = 1.0;
	const double beta = 0.0;

	auto r = m;
	auto rows = m.rows();
	auto cols = m.cols();

	auto nt = cublas::operation::non_transpose;
	auto tr = cublas::operation::transpose;
	cublas::gemm(h, tr, nt, svd.u.rows(), m.cols(), m.rows(),
			&alpha, svd.u.values(), svd.u.rows(),
			m.values(), m.rows(), &beta, r.values(), r.rows());

	auto* svalues = svd.sigma.values();
	auto* rvalues = r.values();
	auto k = [=] __device__ (int tid)
	{
		auto s = svalues[tid % rows];
		rvalues[tid] /= s;
	};
	util::transform<128, 8>(k, rows * cols);

	cublas::gemm(h, tr, nt, svd.vt.rows(), r.cols(), r.rows(),
			&alpha, svd.vt.values(), svd.vt.rows(),
			r.values(), r.rows(), &beta, m.values(), m.rows());
	return m;
}

} // namespace algo

