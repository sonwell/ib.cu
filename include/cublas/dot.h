#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
dot(handle_t h, int n, const float* x, int incx, const float* y, int incy,
		float* result)
{
	throw_if_error(cublasSdot(h, n, x, incx, y, incy, result));
}

inline void
dot(handle_t h, int n, const double* x, int incx, const double* y, int incy,
		double* result)
{
	throw_if_error(cublasDdot(h, n, x, incx, y, incy, result));
}

} // namespace cublas
