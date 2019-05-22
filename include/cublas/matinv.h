#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
matinv_batched(handle_t h, int n, const float* const a[], int lda,
		float* const ai[], int ldai, int* info, int batch_size)
{
	throw_if_error(cublasSmatinvBatched(h, n, a[], lda, ai[], ldai, info,
		batch_size));
}

inline void
matinv_batched(handle_t h, int n, const double* const a[], int lda,
		double* const ai[], int ldai, int* info, int batch_size)
{
	throw_if_error(cublasDmatinvBatched(h, n, a[], lda, ai[], ldai, info,
		batch_size));
}

} // namespace cublas
