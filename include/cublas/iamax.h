#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
iamax(handle_t h, int n, const float* x, int incx, int* result)
{
	throw_if_error(cublasIsamax(h, n, x, incx, result));
}

inline void
iamax(handle_t h, int n, const double* x, int incx, int* result)
{
	throw_if_error(cublasIdamax(h, n, x, incx, result));
}

} // namespace cublas
