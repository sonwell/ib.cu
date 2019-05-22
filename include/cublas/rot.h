#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
rot(handle_t h, int n, float* x, int incx, float* y, int incy, const float* c,
		const float* s)
{
	throw_if_error(cublasSrot(h, n, x, incx, y, incy, c, s));
}

inline void
rot(handle_t h, int n, double* x, int incx, double* y, int incy, const double* c,
		const double* s)
{
	throw_if_error(cublasDrot(h, n, x, incx, y, incy, c, s));
}

} // namespace cublas
