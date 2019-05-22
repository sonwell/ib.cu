#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "solve_policy.h"
#include "solve_analysis_info.h"
#include "operation.h"
#include "csrsv2_info.h"

namespace cusparse {

inline void
csrsv_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseScsrsv_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info));
}

inline void
csrsv_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseDcsrsv_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info));
}

inline void
csrsv_solve(handle& h, operation_adaptor trans_a, int m, const float* alpha,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info,
		const float* f, float* x)
{
	throw_if_error(cusparseScsrsv_solve(h, trans_a, m, alpha, descr_a,
		values_a, starts_a, indices_a, info, f, x));
}

inline void
csrsv_solve(handle& h, operation_adaptor trans_a, int m, const double* alpha,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, solve_analysis_info info,
		const double* f, double* x)
{
	throw_if_error(cusparseDcsrsv_solve(h, trans_a, m, alpha, descr_a,
		values_a, starts_a, indices_a, info, f, x));
}

inline void
csrsv2_zero_pivot(handle& h, csrsv2_info info, int* position)
{
	throw_if_error(cusparseXcsrsv2_zeroPivot(h, info, position));
}

inline void
csrsv2_buffer_size(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, csrsv2_info info, int* buffer_size)
{
	throw_if_error(cusparseScsrsv2_bufferSize(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrsv2_buffer_size(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, csrsv2_info info, int* buffer_size)
{
	throw_if_error(cusparseDcsrsv2_bufferSize(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrsv2_buffer_size_ext(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, csrsv2_info info, size_t* buffer_size)
{
	throw_if_error(cusparseScsrsv2_bufferSizeExt(h, trans_a, m, nnz,
		descr_a, values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrsv2_buffer_size_ext(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, csrsv2_info info, size_t* buffer_size)
{
	throw_if_error(cusparseDcsrsv2_bufferSizeExt(h, trans_a, m, nnz,
		descr_a, values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrsv2_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, csrsv2_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsrsv2_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, policy, buffer));
}

inline void
csrsv2_analysis(handle& h, operation_adaptor trans_a, int m, int nnz,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, csrsv2_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDcsrsv2_analysis(h, trans_a, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, policy, buffer));
}

inline void
csrsv2_solve(handle& h, operation_adaptor trans_a, int m, int nnz,
		const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		csrsv2_info info, const float* f, float* x, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseScsrsv2_solve(h, trans_a, m, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, info, f, x, policy, buffer));
}

inline void
csrsv2_solve(handle& h, operation_adaptor trans_a, int m, int nnz,
		const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		csrsv2_info info, const double* f, double* x,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDcsrsv2_solve(h, trans_a, m, nnz, alpha, descr_a,
		values_a, starts_a, indices_a, info, f, x, policy, buffer));
}

} // namespace cusparse
