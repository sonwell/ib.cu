#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"
#include "operation.h"

namespace cusparse {

inline void
bsrxmv(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int size_of_mask, int mb, int nb, int nnzb, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* mask_a, const int* starts_a, const int* ends_a,
		const int* indices_a, int block_dim, const float* x, const float* beta,
		float* y)
{
	throw_if_error(cusparseSbsrxmv(h, dir_a, trans_a, size_of_mask, mb, nb,
		nnzb, alpha, descr_a, values_a, mask_a, starts_a, ends_a, indices_a,
		block_dim, x, beta, y));
}

inline void
bsrxmv(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int size_of_mask, int mb, int nb, int nnzb, const double* alpha,
		const matrix_description descr_a, const double* values_a,
		const int* mask_a, const int* starts_a, const int* ends_a,
		const int* indices_a, int block_dim, const double* x,
		const double* beta, double* y)
{
	throw_if_error(cusparseDbsrxmv(h, dir_a, trans_a, size_of_mask, mb, nb,
		nnzb, alpha, descr_a, values_a, mask_a, starts_a, ends_a, indices_a,
		block_dim, x, beta, y));
}

} // namespace cusparse
