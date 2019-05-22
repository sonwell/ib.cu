#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "operation.h"

namespace cusparse {

inline void
csrmm(handle& h, operation_adaptor trans_a, int m, int n, int k, int nnz,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const float* b, int ldb, const float* beta, float* c, int ldc)
{
	throw_if_error(cusparseScsrmm(h, trans_a, m, n, k, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, b, ldb, beta, c, ldc));
}

inline void
csrmm(handle& h, operation_adaptor trans_a, int m, int n, int k, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* b, int ldb, const double* beta, double* c, int ldc)
{
	throw_if_error(cusparseDcsrmm(h, trans_a, m, n, k, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, b, ldb, beta, c, ldc));
}

inline void
csrmm2(handle& h, operation_adaptor trans_a, operation_adaptor trans_b, int m,
		int n, int k, int nnz, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* b, int ldb,
		const float* beta, float* c, int ldc)
{
	throw_if_error(cusparseScsrmm2(h, trans_a, trans_b, m, n, k, nnz, alpha,
		descr_a, values_a, starts_a, indices_a, b, ldb, beta, c, ldc));
}

inline void
csrmm2(handle& h, operation_adaptor trans_a, operation_adaptor trans_b, int m,
		int n, int k, int nnz, const double* alpha,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* b, int ldb,
		const double* beta, double* c, int ldc)
{
	throw_if_error(cusparseDcsrmm2(h, trans_a, trans_b, m, n, k, nnz, alpha,
		descr_a, values_a, starts_a, indices_a, b, ldb, beta, c, ldc));
}

} // namespace cusparse
