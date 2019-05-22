#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
symv(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha,
		const float* a, int lda, const float* x, int incx, const float* beta,
		float* y, int incy)
{
	throw_if_error(cublasSsymv(h, uplo, n, alpha, a, lda, x, incx, beta, y,
		incy));
}

inline void
symv(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* a, int lda, const double* x, int incx, const double* beta,
		double* y, int incy)
{
	throw_if_error(cublasDsymv(h, uplo, n, alpha, a, lda, x, incx, beta, y,
		incy));
}

} // namespace cublas
