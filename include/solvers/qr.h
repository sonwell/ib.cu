#pragma once
#include <utility>
#include "types.h"
#include "util/memory.h"
#include "cuda/stream.h"
#include "cusolver/handle.h"
#include "cusolver/stream.h"
#include "cusolver/operation.h"
#include "cusolver/geqrf.h"
#include "cusolver/ormqr.h"
#include "cublas/handle.h"
#include "cublas/trsm.h"
#include "cublas/side_mode.h"
#include "cublas/diagonal_type.h"
#include "cublas/fill_mode.h"

namespace solvers {

struct qr {
	util::memory<double> tau;
	util::memory<int> info;
	dense::matrix m;

	qr(dense::matrix op) :
		tau(std::min(op.cols(), op.rows())),
		info(1), m(std::move(op))
	{
		int buffer_size;
		cusolver::dense::handle h;
		cuda::stream stream;
		cusolver::set_stream(h, stream);

		cusolver::geqrf_buffer_size(h, m.rows(), m.cols(),
				m.values(), m.rows(), &buffer_size);
		util::memory<double> buffer(buffer_size);

		cusolver::geqrf(h, m.rows(), m.cols(), m.values(),
				m.rows(), tau, buffer, buffer_size, info);
	}
};

template <template <typename> typename container>
container<linalg::dense<double>>
solve(const qr& qr, container<linalg::dense<double>> m)
{
	static constexpr auto alpha = 1.0;
	int buffer_size;
	cusolver::dense::handle h;
	cublas::handle k;
	cuda::stream stream;
	cusolver::set_stream(h, stream);
	int reflections = std::min(qr.m.cols(), qr.m.rows());

	constexpr auto op = cusolver::operation::transpose;
	constexpr auto nt = cusolver::operation::non_transpose;
	constexpr auto side = cublas::side_mode::left;
	constexpr auto fill = cublas::fill_mode::upper;
	constexpr auto diag = cublas::diagonal_type::non_unit;
	cusolver::ormqr_buffer_size(h, side, op, m.rows(), m.cols(), reflections,
			qr.m.values(), qr.m.rows(), qr.tau, m.values(), m.rows(),
			&buffer_size);
	util::memory<double> buffer(buffer_size);

	cusolver::ormqr(h, side, op, m.rows(), m.cols(), reflections,
			qr.m.values(), qr.m.rows(), qr.tau, m.values(), m.rows(),
			buffer, buffer_size, qr.info);
	cublas::trsm(k, side, fill, nt, diag, qr.m.cols(), m.cols(),
			&alpha, qr.m.values(), qr.m.rows(), m.values(), m.rows());
	if (qr.m.rows() == qr.m.cols())
		return m;
	auto rows = qr.m.cols();
	auto cols = m.cols();
	auto ld = m.rows();
	linalg::size sz{rows, cols};
	container<linalg::dense<double>> r{sz};
	auto* src = m.values();
	auto* dst = r.values();
	auto copy = [=] __device__ (int tid)
	{
		auto r = tid % rows;
		auto c = tid / rows;
		dst[tid] = src[r + ld * c];
	};
	util::transform<128, 7>(copy, rows * cols);
	return r;
}

} // namespace solvers
