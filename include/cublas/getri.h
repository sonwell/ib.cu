#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
getri_batched(handle_t h, int n, const float* const a[], int lda,
		const int* pivots, float* const c[], int ldc, int* info, int batch_size)
{
	throw_if_error(cublasSgetriBatched(h, n, a, lda, pivots, c, ldc,
		info, batch_size));
}

inline void
getri_batched(handle_t h, int n, const double* const a[], int lda,
		const int* pivots, double* const c[], int ldc, int* info,
		int batch_size)
{
	throw_if_error(cublasDgetriBatched(h, n, a, lda, pivots, c, ldc,
		info, batch_size));
}

} // namespace cublas
