#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
geqrf_batched(handle_t h, int m, int n, float* const a[], int lda,
		float* const tau[], int* info, int batch_size)
{
	throw_if_error(cublasSgeqrfBatched(h, m, n, a[], lda, tau[], info,
		batch_size));
}

inline void
geqrf_batched(handle_t h, int m, int n, double* const a[], int lda,
		double* const tau[], int* info, int batch_size)
{
	throw_if_error(cublasDgeqrfBatched(h, m, n, a[], lda, tau[], info,
		batch_size));
}

} // namespace cublas
