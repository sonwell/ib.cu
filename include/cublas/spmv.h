#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
spmv(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha,
		const float* a, const float* x, int incx, const float* beta, float* y,
		int incy)
{
	throw_if_error(cublasSspmv(h, uplo, n, alpha, a, x, incx, beta, y, incy));
}

inline void
spmv(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* a, const double* x, int incx, const double* beta,
		double* y, int incy)
{
	throw_if_error(cublasDspmv(h, uplo, n, alpha, a, x, incx, beta, y, incy));
}

} // namespace cublas
