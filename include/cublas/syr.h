#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
syr(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha, const float* x,
		int incx, float* a, int lda)
{
	throw_if_error(cublasSsyr(h, uplo, n, alpha, x, incx, a, lda));
}

inline void
syr(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* x, int incx, double* a, int lda)
{
	throw_if_error(cublasDsyr(h, uplo, n, alpha, x, incx, a, lda));
}

inline void
syr2(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha,
		const float* x, int incx, const float* y, int incy, float* a, int lda)
{
	throw_if_error(cublasSsyr2(h, uplo, n, alpha, x, incx, y, incy, a, lda));
}

inline void
syr2(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* x, int incx, const double* y, int incy, double* a,
		int lda)
{
	throw_if_error(cublasDsyr2(h, uplo, n, alpha, x, incx, y, incy, a, lda));
}

} // namespace cublas
