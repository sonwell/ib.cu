#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"
#include "operation.h"

namespace cusparse {

inline void
gemvi(handle& h, operation_adaptor trans_a, int m, int n, const float* alpha,
		const float* a, int lda, int nnz, const float* values,
		const int* indices, const float* beta, float* y,
		index_base_adaptor base, void* buffer)
{
	throw_if_error(cusparseSgemvi(h, trans_a, m, n, alpha, a, lda, nnz,
		values, indices, beta, y, base, buffer));
}

inline void
gemvi(handle& h, operation_adaptor trans_a, int m, int n, const double* alpha,
		const double* a, int lda, int nnz, const double* values,
		const int* indices, const double* beta, double* y,
		index_base_adaptor base, void* buffer)
{
	throw_if_error(cusparseDgemvi(h, trans_a, m, n, alpha, a, lda, nnz,
		values, indices, beta, y, base, buffer));
}

inline void
gemvi_buffer_size(handle& h, operation_adaptor trans_a, int m, int n, int nnz,
		int* buffer_size)
{
	throw_if_error(cusparseSgemvi_bufferSize(h, trans_a, m, n, nnz,
		buffer_size));
}

} // namespace cusparse
