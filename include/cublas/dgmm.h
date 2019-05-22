#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "side_mode.h"

namespace cublas {

inline void
dgmm(handle_t h, side_mode_adaptor mode, int m, int n, const float* a, int lda,
		const float* x, int incx, float* c, int ldc)
{
	throw_if_error(cublasSdgmm(h, mode, m, n, a, lda, x, incx, c, ldc));
}

inline void
dgmm(handle_t h, side_mode_adaptor mode, int m, int n, const double* a, int lda,
		const double* x, int incx, double* c, int ldc)
{
	throw_if_error(cublasDdgmm(h, mode, m, n, a, lda, x, incx, c, ldc));
}

} // namespace cublas
