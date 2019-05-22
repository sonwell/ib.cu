#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
scal(handle_t h, int n, const float* alpha, float* x, int incx)
{
	throw_if_error(cublasSscal(h, n, alpha, x, incx));
}

inline void
scal(handle_t h, int n, const double* alpha, double* x, int incx)
{
	throw_if_error(cublasDscal(h, n, alpha, x, incx));
}

} // namespace cublas
