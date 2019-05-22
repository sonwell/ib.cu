#pragma once
#include "types.h"
#include "exceptions.h"

namespace cublas {

inline void
copy(handle_t h, int n, const float* x, int incx, float* y, int incy)
{
	throw_if_error(cublasScopy(h, n, x, incx, y, incy));
}

inline void
copy(handle_t h, int n, const double* x, int incx, double* y, int incy)
{
	throw_if_error(cublasDcopy(h, n, x, incx, y, incy));
}

} // namespace cublas
