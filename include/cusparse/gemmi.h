#pragma once
#include "types.h"
#include "handle.h"

namespace cusparse {

inline void
gemmi(handle& h, int m, int n, int k, int nnz, const float* alpha,
		const float* a, int lda, const float* values_b,
		const int* startsvalues_b, const float* beta, float* c, int ldc)
{
	throw_if_error(cusparseSgemmi(h, m, n, k, nnz, alpha, a, lda, values_b,
		startsvalues_b, beta, c, ldc));
}

inline void
gemmi(handle& h, int m, int n, int k, int nnz, const double* alpha,
		const double* a, int lda, const double* values_b,
		const int* startsvalues_b, const double* beta, double* c, int ldc)
{
	throw_if_error(cusparseDgemmi(h, m, n, k, nnz, alpha, a, lda, values_b,
		startsvalues_b, beta, c, ldc));
}

} // namespace cusparse