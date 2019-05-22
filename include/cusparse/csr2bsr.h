#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
csr2bsr_nnz(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const int* starts_a,
		const int* indices_a, int block_dim, const matrix_description descr_c,
		int* starts_c, int* nnz_total)
{
	throw_if_error(cusparseXcsr2bsrNnz(h, dir_a, m, n, descr_a, starts_a,
		indices_a, block_dim, descr_c, starts_c, nnz_total));
}

inline void
csr2bsr(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int block_dim,
		const matrix_description descr_c, float* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseScsr2bsr(h, dir_a, m, n, descr_a, values_a,
		starts_a, indices_a, block_dim, descr_c, values_c, starts_c,
		indices_c));
}

inline void
csr2bsr(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int block_dim,
		const matrix_description descr_c, double* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseDcsr2bsr(h, dir_a, m, n, descr_a, values_a,
		starts_a, indices_a, block_dim, descr_c, values_c, starts_c,
		indices_c));
}

} // namespace cusparse
