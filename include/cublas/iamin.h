#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
iamin(handle_t h, int n, const float* x, int incx, int* result)
{
	throw_if_error(cublasIsamin(h, n, x, incx, result));
}

inline void
iamin(handle_t h, int n, const double* x, int incx, int* result)
{
	throw_if_error(cublasIdamin(h, n, x, incx, result));
}

} // namespace cublas
