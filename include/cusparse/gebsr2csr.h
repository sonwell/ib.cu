#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
gebsr2csr(handle& h, direction_adaptor dir_a, int mb, int nb,
		const matrix_description descr_a, const int* starts_a,
		const int* indices_a, int row_block_dim, int col_block_dim,
		const matrix_description descr_c, int* starts_c, int* indices_c)
{
	throw_if_error(cusparseXgebsr2csr(h, dir_a, mb, nb, descr_a, starts_a,
		indices_a, row_block_dim, col_block_dim, descr_c, starts_c,
		indices_c));
}

inline void
gebsr2csr(handle& h, direction_adaptor dir_a, int mb, int nb,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, const matrix_description descr_c, float* values_c,
		int* starts_c, int* indices_c)
{
	throw_if_error(cusparseSgebsr2csr(h, dir_a, mb, nb, descr_a, values_a,
		starts_a, indices_a, row_block_dim, col_block_dim, descr_c,
		values_c, starts_c, indices_c));
}

inline void
gebsr2csr(handle& h, direction_adaptor dir_a, int mb, int nb,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, const matrix_description descr_c, double* values_c,
		int* starts_c, int* indices_c)
{
	throw_if_error(cusparseDgebsr2csr(h, dir_a, mb, nb, descr_a, values_a,
		starts_a, indices_a, row_block_dim, col_block_dim, descr_c,
		values_c, starts_c, indices_c));
}

} // namespace cusparse
