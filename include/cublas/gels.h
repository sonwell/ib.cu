#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"

namespace cublas {

inline void
gels_batched(handle_t h, operation_adaptor trans, int m, int n, int nrhs,
		float* const a[], int lda, float* const c[], int ldc, int* info,
		int* info_array, int batch_size)
{
	throw_if_error(cublasSgelsBatched(h, trans, m, n, nrhs, a[], lda, c[],
		ldc, info, info_array, batch_size));
}

inline void
gels_batched(handle_t h, operation_adaptor trans, int m, int n, int nrhs,
		double* const a[], int lda, double* const c[], int ldc, int* info,
		int* info_array, int batch_size)
{
	throw_if_error(cublasDgelsBatched(h, trans, m, n, nrhs, a[], lda, c[],
		ldc, info, info_array, batch_size));
}

} // namespace cublas
