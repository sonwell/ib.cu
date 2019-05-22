#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"
#include "operation.h"

namespace cusparse {

inline void
bsrmv(handle& h, direction_adaptor dir_a, operation_adaptor trans_a, int mb,
		int nb, int nnzb, const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		int block_dim, const float* x, const float* beta, float* y)
{
	throw_if_error(cusparseSbsrmv(h, dir_a, trans_a, mb, nb, nnzb, alpha,
		descr_a, values_a, starts_a, indices_a, block_dim, x, beta, y));
}

inline void
bsrmv(handle& h, direction_adaptor dir_a, operation_adaptor trans_a, int mb,
		int nb, int nnzb, const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		int block_dim, const double* x, const double* beta, double* y)
{
	throw_if_error(cusparseDbsrmv(h, dir_a, trans_a, mb, nb, nnzb, alpha,
		descr_a, values_a, starts_a, indices_a, block_dim, x, beta, y));
}

} // namespace cusparse
