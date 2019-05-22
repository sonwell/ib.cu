#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"

namespace cublas {

inline void
getrs_batched(handle_t h, operation_adaptor trans, int n, int nrhs,
		const float* const a[], int lda, const int* pivots, float* const b[],
		int ldb, int* info, int batch_size)
{
	throw_if_error(cublasSgetrsBatched(h, trans, n, nrhs, a, lda, pivots, b,
		ldb, info, batch_size));
}

inline void
getrs_batched(handle_t h, operation_adaptor trans, int n, int nrhs,
		const double* const a[], int lda, const int* pivots, double* const b[],
		int ldb, int* info, int batch_size)
{
	throw_if_error(cublasDgetrsBatched(h, trans, n, nrhs, a, lda, pivots, b,
		ldb, info, batch_size));
}

} // namespace cublas
