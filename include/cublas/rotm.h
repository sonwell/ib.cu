#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
rotm(handle_t h, int n, float* x, int incx, float* y, int incy,
		const float* param)
{
	throw_if_error(cublasSrotm(h, n, x, incx, y, incy, param));
}

inline void
rotm(handle_t h, int n, double* x, int incx, double* y, int incy,
		const double* param)
{
	throw_if_error(cublasDrotm(h, n, x, incx, y, incy, param));
}

} // namespace cublas
