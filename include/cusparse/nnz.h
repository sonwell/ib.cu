#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
nnz(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const float* a, int lda, int* nnz_per,
		int* nnz_total)
{
	throw_if_error(cusparseSnnz(h, dir_a, m, n, descr_a, a, lda, nnz_per,
		nnz_total));
}

inline void
nnz(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const double* a, int lda,
		int* nnz_per, int* nnz_total)
{
	throw_if_error(cusparseDnnz(h, dir_a, m, n, descr_a, a, lda, nnz_per,
		nnz_total));
}

inline void
nnz_compress(handle& h, int m, const matrix_description descr,
		const float* values_a, const int* starts_a, int* nnz_per_row,
		int* nnz_c, float tol)
{
	throw_if_error(cusparseSnnz_compress(h, m, descr, values_a, starts_a,
		nnz_per_row, nnz_c, tol));
}

inline void
nnz_compress(handle& h, int m, const matrix_description descr,
		const double* values_a, const int* starts_a, int* nnz_per_row,
		int* nnz_c, float tol)
{
	throw_if_error(cusparseDnnz_compress(h, m, descr, values_a, starts_a,
		nnz_per_row, nnz_c, tol));
}

} // namespace cusparse
