#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusparse {

inline void
dense2csc(handle& h, int m, int n, const matrix_description descr_a,
		const float* a, int lda, const int* nnz_per_col, float* values_a,
		int* starts_a, int* indices_a)
{
	throw_if_error(cusparseSdense2csc(h, m, n, descr_a, a, lda, nnz_per_col,
		values_a, starts_a, indices_a));
}

inline void
dense2csc(handle& h, int m, int n, const matrix_description descr_a,
		const double* a, int lda, const int* nnz_per_col, double* values_a,
		int* starts_a, int* indices_a)
{
	throw_if_error(cusparseDdense2csc(h, m, n, descr_a, a, lda, nnz_per_col,
		values_a, starts_a, indices_a));
}

} // namespace cusparse
