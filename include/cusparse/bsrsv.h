#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "solve_policy.h"
#include "direction.h"
#include "bsrsv2_info.h"
#include "operation.h"

namespace cusparse {

inline void
bsrsv2_zero_pivot(handle& h, bsrsv2_info info, int* position)
{
	throw_if_error(cusparseXbsrsv2_zeroPivot(h, info, position));
}

inline void
bsrsv2_buffer_size(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, int mb, int nnzb,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, int block_dim, bsrsv2_info info, int* buffer_size)
{
	throw_if_error(cusparseSbsrsv2_bufferSize(h, dir_a, trans_a, mb, nnzb,
		descr_a, values_a, starts_a, indices_a, block_dim, info,
		buffer_size));
}

inline void
bsrsv2_buffer_size(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, int mb, int nnzb,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, int block_dim, bsrsv2_info info, int* buffer_size)
{
	throw_if_error(cusparseDbsrsv2_bufferSize(h, dir_a, trans_a, mb, nnzb,
		descr_a, values_a, starts_a, indices_a, block_dim, info,
		buffer_size));
}

inline void
bsrsv2_buffer_size_ext(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, int mb, int nnzb,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, int block_size, bsrsv2_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSbsrsv2_bufferSizeExt(h, dir_a, trans_a, mb,
		nnzb, descr_a, values_a, starts_a, indices_a, block_size, info,
		buffer_size));
}

inline void
bsrsv2_buffer_size_ext(handle& h, direction_adaptor dir_a,
		operation_adaptor trans_a, int mb, int nnzb,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, int block_size, bsrsv2_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDbsrsv2_bufferSizeExt(h, dir_a, trans_a, mb,
		nnzb, descr_a, values_a, starts_a, indices_a, block_size, info,
		buffer_size));
}

inline void
bsrsv2_analysis(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int mb, int nnzb, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		int block_dim, bsrsv2_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseSbsrsv2_analysis(h, dir_a, trans_a, mb, nnzb,
		descr_a, values_a, starts_a, indices_a, block_dim, info, policy,
		buffer));
}

inline void
bsrsv2_analysis(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int mb, int nnzb, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		int block_dim, bsrsv2_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseDbsrsv2_analysis(h, dir_a, trans_a, mb, nnzb,
		descr_a, values_a, starts_a, indices_a, block_dim, info, policy,
		buffer));
}

inline void
bsrsv2_solve(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int mb, int nnzb, const float* alpha, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		int block_dim, bsrsv2_info info, const float* f, float* x,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsrsv2_solve(h, dir_a, trans_a, mb, nnzb, alpha,
		descr_a, values_a, starts_a, indices_a, block_dim, info, f, x,
		policy, buffer));
}

inline void
bsrsv2_solve(handle& h, direction_adaptor dir_a, operation_adaptor trans_a,
		int mb, int nnzb, const double* alpha, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		int block_dim, bsrsv2_info info, const double* f, double* x,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsrsv2_solve(h, dir_a, trans_a, mb, nnzb, alpha,
		descr_a, values_a, starts_a, indices_a, block_dim, info, f, x,
		policy, buffer));
}

} // namespace cusparse
