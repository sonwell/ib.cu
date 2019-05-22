#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusparse {

inline void
csr2csr_compress(handle& h, int m, int n, const matrix_description descr_a,
		const float* values_a, const int* indices_a, const int* starts_a,
		int nnz_a, const int* nnz_per_row, float* values_c, int* indices_c,
		int* starts_c, float tol)
{
	throw_if_error(cusparseScsr2csr_compress(h, m, n, descr_a, values_a,
		indices_a, starts_a, nnz_a, nnz_per_row, values_c, indices_c,
		starts_c, tol));
}

inline void
csr2csr_compress(handle& h, int m, int n, const matrix_description descr_a,
		const double* values_a, const int* indices_a, const int* starts_a,
		int nnz_a, const int* nnz_per_row, double* values_c, int* indices_c,
		int* starts_c, float tol)
{
	throw_if_error(cusparseDcsr2csr_compress(h, m, n, descr_a, values_a,
		indices_a, starts_a, nnz_a, nnz_per_row, values_c, indices_c,
		starts_c, tol));
}

} // namespace cusparse
