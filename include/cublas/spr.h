#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
spr(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha, const float* x,
		int incx, float* a)
{
	throw_if_error(cublasSspr(h, uplo, n, alpha, x, incx, a));
}

inline void
spr(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* x, int incx, double* a)
{
	throw_if_error(cublasDspr(h, uplo, n, alpha, x, incx, a));
}

inline void
spr2(handle_t h, fill_mode_adaptor uplo, int n, const float* alpha,
		const float* x, int incx, const float* y, int incy, float* a)
{
	throw_if_error(cublasSspr2(h, uplo, n, alpha, x, incx, y, incy, a));
}

inline void
spr2(handle_t h, fill_mode_adaptor uplo, int n, const double* alpha,
		const double* x, int incx, const double* y, int incy, double* a)
{
	throw_if_error(cublasDspr2(h, uplo, n, alpha, x, incx, y, incy, a));
}

} // namespace cublas
