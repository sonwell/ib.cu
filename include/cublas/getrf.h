#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
getrf_batched(handle_t h, int n, float* const a[], int lda, int* pivots,
		int* info, int batch_size)
{
	throw_if_error(cublasSgetrfBatched(h, n, a, lda, pivots, info,
		batch_size));
}

inline void
getrf_batched(handle_t h, int n, double* const a[], int lda, int* pivots,
		int* info, int batch_size)
{
	throw_if_error(cublasDgetrfBatched(h, n, a, lda, pivots, info,
		batch_size));
}

} // namespace cublas
