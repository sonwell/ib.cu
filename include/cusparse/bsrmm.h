#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"
#include "operation.h"

namespace cusparse {

inline void
bsrmm(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int kb, int nnzb,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const int block_size, const float* b, const int ldb, const float* beta,
		float* c, int ldc)
{
	throw_if_error(cusparseSbsrmm(h, dir_a, trans_a, trans_b, mb, n, kb,
		nnzb, alpha, descr_a, values_a, starts_a, indices_a, block_size, b,
		ldb, beta, c, ldc));
}

inline void
bsrmm(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int kb, int nnzb,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const int block_size, const double* b, const int ldb,
		const double* beta, double* c, int ldc)
{
	throw_if_error(cusparseDbsrmm(h, dir_a, trans_a, trans_b, mb, n, kb,
		nnzb, alpha, descr_a, values_a, starts_a, indices_a, block_size, b,
		ldb, beta, c, ldc));
}

} // namespace cusparse
