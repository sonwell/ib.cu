#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"

namespace cublas {

inline void
gemv(handle_t h, operation_adaptor trans, int m, int n, const float* alpha,
		const float* a, int lda, const float* x, int incx, const float* beta,
		float* y, int incy)
{
	throw_if_error(cublasSgemv(h, trans, m, n, alpha, a, lda, x, incx, beta,
		y, incy));
}

inline void
gemv(handle_t h, operation_adaptor trans, int m, int n, const double* alpha,
		const double* a, int lda, const double* x, int incx, const double* beta,
		double* y, int incy)
{
	throw_if_error(cublasDgemv(h, trans, m, n, alpha, a, lda, x, incx, beta,
		y, incy));
}

} // namespace cublas
