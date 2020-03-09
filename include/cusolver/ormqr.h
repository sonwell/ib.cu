#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "cublas/side_mode.h"

namespace cusolver {

inline void
ormqr_buffer_size(dense::handle_t h, cublas::side_mode_adaptor side,
		operation_adaptor trans, int m, int n, int k, const float* a, int lda,
		const float* tau, const float* c, int ldc, int* work_size)
{
	throw_if_error(cusolverDnSormqr_bufferSize(h, side, trans, m, n, k, a,
		lda, tau, c, ldc, work_size));
}

inline void
ormqr_buffer_size(dense::handle_t h, cublas::side_mode_adaptor side,
		operation_adaptor trans, int m, int n, int k, const double* a, int lda,
		const double* tau, const double* c, int ldc, int* work_size)
{
	throw_if_error(cusolverDnDormqr_bufferSize(h, side, trans, m, n, k, a,
		lda, tau, c, ldc, work_size));
}

inline void
ormqr(dense::handle_t h, cublas::side_mode_adaptor side, operation_adaptor trans,
		int m, int n, int k, const float* a, int lda, const float* tau, float* c,
		int ldc, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSormqr(h, side, trans, m, n, k, a, lda, tau, c,
		ldc, work, work_size, info));
}

inline void
ormqr(dense::handle_t h, cublas::side_mode_adaptor side, operation_adaptor trans,
		int m, int n, int k, const double* a, int lda, const double* tau, double* c,
		int ldc, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDormqr(h, side, trans, m, n, k, a, lda, tau, c,
		ldc, work, work_size, info));
}

} // namespace cusolver
