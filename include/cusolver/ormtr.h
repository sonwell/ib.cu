#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "fill_mode.h"
#include "side_mode.h"

namespace cusolver {

inline void
ormtr_buffer_size(dense::handle_t h, side_mode_adaptor side,
		fill_mode_adaptor uplo, operation_adaptor trans, int m, int n,
		const float* a, int lda, const float* tau, const float* c, int ldc,
		int* work_size)
{
	throw_if_error(cusolverDnSormtr_bufferSize(h, side, uplo, trans, m, n,
		a, lda, tau, c, ldc, work_size));
}

inline void
ormtr_buffer_size(dense::handle_t h, side_mode_adaptor side,
		fill_mode_adaptor uplo, operation_adaptor trans, int m, int n,
		const double* a, int lda, const double* tau, const double* c, int ldc,
		int* work_size)
{
	throw_if_error(cusolverDnDormtr_bufferSize(h, side, uplo, trans, m, n,
		a, lda, tau, c, ldc, work_size));
}

inline void
ormtr(dense::handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, int m, int n, float* a, int lda, float* tau,
		float* c, int ldc, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSormtr(h, side, uplo, trans, m, n, a, lda, tau,
		c, ldc, work, work_size, info));
}

inline void
ormtr(dense::handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, int m, int n, double* a, int lda, double* tau,
		double* c, int ldc, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDormtr(h, side, uplo, trans, m, n, a, lda, tau,
		c, ldc, work, work_size, info));
}

} // namespace cusolver
