#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
swap(handle_t h, int n, float* x, int incx, float* y, int incy)
{
	throw_if_error(cublasSswap(h, n, x, incx, y, incy));
}

inline void
swap(handle_t h, int n, double* x, int incx, double* y, int incy)
{
	throw_if_error(cublasDswap(h, n, x, incx, y, incy));
}

} // namespace cublas
