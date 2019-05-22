#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
asum(handle_t h, int n, const float* x, int incx, float* result)
{
	throw_if_error(cublasSasum(h, n, x, incx, result));
}

inline void
asum(handle_t h, int n, const double* x, int incx, double* result)
{
	throw_if_error(cublasDasum(h, n, x, incx, result));
}

} // namespace cublas
