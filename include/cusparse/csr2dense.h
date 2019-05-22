#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusparse {

inline void
csr2dense(handle& h, int m, int n, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		float* a, int lda)
{
	throw_if_error(cusparseScsr2dense(h, m, n, descr_a, values_a, starts_a,
		indices_a, a, lda));
}

inline void
csr2dense(handle& h, int m, int n, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		double* a, int lda)
{
	throw_if_error(cusparseDcsr2dense(h, m, n, descr_a, values_a, starts_a,
		indices_a, a, lda));
}

} // namespace cusparse
