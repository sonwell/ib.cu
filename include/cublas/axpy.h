#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
axpy(handle_t h, int n, const float* alpha, const float* x, int incx, float* y,
		int incy)
{
	throw_if_error(cublasSaxpy(h, n, alpha, x, incx, y, incy));
}

inline void
axpy(handle_t h, int n, const double* alpha, const double* x, int incx, double* y,
		int incy)
{
	throw_if_error(cublasDaxpy(h, n, alpha, x, incx, y, incy));
}

} // namespace cublas
