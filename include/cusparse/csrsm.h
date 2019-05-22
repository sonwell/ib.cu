#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "csrsm2_info.h"
#include "solve_policy.h"
#include "solve_analysis_info.h"
#include "operation.h"

namespace cusparse {

inline void
csrsm_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseScsrsm_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info));
}

inline void
csrsm_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseDcsrsm_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info));
}

inline void
csrsm_solve(handle& h, operation_adaptor trans_a, int m, int n,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		solve_analysis_info info, const float* b, int ldb, float* x, int ldx)
{
	throw_if_error(cusparseScsrsm_solve(h, trans_a, m, n, alpha, descr_a,
		values_a, starts_a, indices_a, info, b, ldb, x, ldx));
}

inline void
csrsm_solve(handle& h, operation_adaptor trans_a, int m, int n,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		solve_analysis_info info, const double* b, int ldb, double* x, int ldx)
{
	throw_if_error(cusparseDcsrsm_solve(h, trans_a, m, n, alpha, descr_a,
		values_a, starts_a, indices_a, info, b, ldb, x, ldx));
}

inline void
csrsm2_zero_pivot(handle& h, csrsm2_info info, int* position)
{
	throw_if_error(cusparseXcsrsm2_zeroPivot(h, info, position));
}

inline void
csrsm2_buffer_size_ext(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* b, int ldb,
		csrsm2_info info, solve_policy_adaptor policy, size_t* buffer_size)
{
	throw_if_error(cusparseScsrsm2_bufferSizeExt(h, algo, trans_a, trans_b,
		m, nrhs, nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb,
		info, policy, buffer_size));
}

inline void
csrsm2_buffer_size_ext(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* b, int ldb, csrsm2_info info, solve_policy_adaptor policy,
		size_t* buffer_size)
{
	throw_if_error(cusparseDcsrsm2_bufferSizeExt(h, algo, trans_a, trans_b,
		m, nrhs, nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb,
		info, policy, buffer_size));
}

inline void
csrsm2_analysis(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* b, int ldb,
		csrsm2_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsrsm2_analysis(h, algo, trans_a, trans_b, m,
		nrhs, nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb,
		info, policy, buffer));
}

inline void
csrsm2_analysis(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* b, int ldb, csrsm2_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseDcsrsm2_analysis(h, algo, trans_a, trans_b, m,
		nrhs, nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb,
		info, policy, buffer));
}

inline void
csrsm2_solve(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, float* b, int ldb,
		csrsm2_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsrsm2_solve(h, algo, trans_a, trans_b, m, nrhs,
		nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb, info,
		policy, buffer));
}

inline void
csrsm2_solve(handle& h, int algo, operation_adaptor trans_a,
		operation_adaptor trans_b, int m, int nrhs, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		double* b, int ldb, csrsm2_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseDcsrsm2_solve(h, algo, trans_a, trans_b, m, nrhs,
		nnz, alpha, descr_a, values_a, starts_a, indices_a, b, ldb, info,
		policy, buffer));
}

} // namespace cusparse
