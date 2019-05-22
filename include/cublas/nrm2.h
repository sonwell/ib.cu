#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
nrm2(handle_t h, int n, const float* x, int incx, float* result)
{
	throw_if_error(cublasSnrm2(h, n, x, incx, result));
}

inline void
nrm2(handle_t h, int n, const double* x, int incx, double* result)
{
	throw_if_error(cublasDnrm2(h, n, x, incx, result));
}

} // namespace cublas
