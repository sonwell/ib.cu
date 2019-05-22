#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "operation.h"

namespace cusparse {

inline void
csrmv(handle& h, operation_adaptor trans_a, int m, int n, int nnz,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const float* x, const float* beta, float* y)
{
	throw_if_error(cusparseScsrmv(h, trans_a, m, n, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, x, beta, y));
}

inline void
csrmv(handle& h, operation_adaptor trans_a, int m, int n, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* x, const double* beta, double* y)
{
	throw_if_error(cusparseDcsrmv(h, trans_a, m, n, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, x, beta, y), "csrmv failed");
}

inline void
csrmv_mp(handle& h, operation_adaptor trans_a, int m, int n, int nnz,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const float* x, const float* beta, float* y)
{
	throw_if_error(cusparseScsrmv_mp(h, trans_a, m, n, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, x, beta, y));
}

inline void
csrmv_mp(handle& h, operation_adaptor trans_a, int m, int n, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* x, const double* beta, double* y)
{
	throw_if_error(cusparseDcsrmv_mp(h, trans_a, m, n, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, x, beta, y));
}

} // namespace cusparse
