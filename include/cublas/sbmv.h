#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
sbmv(handle_t h, fill_mode_adaptor uplo, int n, int k, const float* alpha,
		const float* a, int lda, const float* x, int incx, const float* beta,
		float* y, int incy)
{
	throw_if_error(cublasSsbmv(h, uplo, n, k, alpha, a, lda, x, incx, beta,
		y, incy));
}

inline void
sbmv(handle_t h, fill_mode_adaptor uplo, int n, int k, const double* alpha,
		const double* a, int lda, const double* x, int incx, const double* beta,
		double* y, int incy)
{
	throw_if_error(cublasDsbmv(h, uplo, n, k, alpha, a, lda, x, incx, beta,
		y, incy));
}

} // namespace cublas
