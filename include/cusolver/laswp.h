#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

inline void
laswp(dense::handle_t h, int n, float* a, int lda, int k1, int k2,
		const int* pivots, int incx)
{
	throw_if_error(cusolverDnSlaswp(h, n, a, lda, k1, k2, pivots, incx));
}

inline void
laswp(dense::handle_t h, int n, double* a, int lda, int k1, int k2,
		const int* pivots, int incx)
{
	throw_if_error(cusolverDnDlaswp(h, n, a, lda, k1, k2, pivots, incx));
}

} // namespace cusolver
