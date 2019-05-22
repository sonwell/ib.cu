#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
ger(handle_t h, int m, int n, const float* alpha, const float* x, int incx,
		const float* y, int incy, float* a, int lda)
{
	throw_if_error(cublasSger(h, m, n, alpha, x, incx, y, incy, a, lda));
}

inline void
ger(handle_t h, int m, int n, const double* alpha, const double* x, int incx,
		const double* y, int incy, double* a, int lda)
{
	throw_if_error(cublasDger(h, m, n, alpha, x, incx, y, incy, a, lda));
}

} // namespace cublas
