#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "bsrsm2_info.h"
#include "solve_policy.h"
#include "direction.h"
#include "operation.h"

namespace cusparse {

inline void
bsrsm2_zero_pivot(handle& h, bsrsm2_info info, int* position)
{
	throw_if_error(cusparseXbsrsm2_zeroPivot(h, info, position));
}

inline void
bsrsm2_buffer_size(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, operation_adaptor trans_b, int mb, int n,
		int nnzb, const matrix_description descr_a, float* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		int* buffer_size)
{
	throw_if_error(cusparseSbsrsm2_bufferSize(h, dir_a, trans_a, trans_b,
		mb, n, nnzb, descr_a, values, starts, indices, block_size, info,
		buffer_size));
}

inline void
bsrsm2_buffer_size(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, operation_adaptor trans_b, int mb, int n,
		int nnzb, const matrix_description descr_a, double* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		int* buffer_size)
{
	throw_if_error(cusparseDbsrsm2_bufferSize(h, dir_a, trans_a, trans_b,
		mb, n, nnzb, descr_a, values, starts, indices, block_size, info,
		buffer_size));
}

inline void
bsrsm2_buffer_size_ext(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, operation_adaptor trans_b, int mb, int n,
		int nnzb, const matrix_description descr_a, float* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSbsrsm2_bufferSizeExt(h, dir_a, trans_a, trans_b,
		mb, n, nnzb, descr_a, values, starts, indices, block_size, info,
		buffer_size));
}

inline void
bsrsm2_buffer_size_ext(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, operation_adaptor trans_b, int mb, int n,
		int nnzb, const matrix_description descr_a, double* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDbsrsm2_bufferSizeExt(h, dir_a, trans_a, trans_b,
		mb, n, nnzb, descr_a, values, starts, indices, block_size, info,
		buffer_size));
}

inline void
bsrsm2_analysis(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int nnzb,
		const matrix_description descr_a, const float* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsrsm2_analysis(h, dir_a, trans_a, trans_b, mb,
		n, nnzb, descr_a, values, starts, indices, block_size, info, policy,
		buffer));
}

inline void
bsrsm2_analysis(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int nnzb,
		const matrix_description descr_a, const double* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsrsm2_analysis(h, dir_a, trans_a, trans_b, mb,
		n, nnzb, descr_a, values, starts, indices, block_size, info, policy,
		buffer));
}

inline void
bsrsm2_solve(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int nnzb, const float* alpha,
		const matrix_description descr_a, const float* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		const float* b, int ldb, float* x, int ldx, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseSbsrsm2_solve(h, dir_a, trans_a, trans_b, mb, n,
		nnzb, alpha, descr_a, values, starts, indices, block_size, info, b,
		ldb, x, ldx, policy, buffer));
}

inline void
bsrsm2_solve(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		operation_adaptor trans_b, int mb, int n, int nnzb, const double* alpha,
		const matrix_description descr_a, const double* values,
		const int* starts, const int* indices, int block_size, bsrsm2_info info,
		const double* b, int ldb, double* x, int ldx,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsrsm2_solve(h, dir_a, trans_a, trans_b, mb, n,
		nnzb, alpha, descr_a, values, starts, indices, block_size, info, b,
		ldb, x, ldx, policy, buffer));
}

} // namespace cusparse
