#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "bsrilu02_info.h"
#include "solve_policy.h"
#include "direction.h"

namespace cusparse {

inline void
bsrilu02_numeric_boost(handle& h, bsrilu02_info info, int enable_boost,
		double* tol, float* boost_val)
{
	throw_if_error(cusparseSbsrilu02_numericBoost(h, info, enable_boost,
		tol, boost_val));
}

inline void
bsrilu02_numeric_boost(handle& h, bsrilu02_info info, int enable_boost,
		double* tol, double* boost_val)
{
	throw_if_error(cusparseDbsrilu02_numericBoost(h, info, enable_boost,
		tol, boost_val));
}

inline void
bsrilu02_zero_pivot(handle& h, bsrilu02_info info, int* position)
{
	throw_if_error(cusparseXbsrilu02_zeroPivot(h, info, position));
}

inline void
bsrilu02_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info, int* buffer_size)
{
	throw_if_error(cusparseSbsrilu02_bufferSize(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, buffer_size));
}

inline void
bsrilu02_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info, int* buffer_size)
{
	throw_if_error(cusparseDbsrilu02_bufferSize(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, buffer_size));
}

inline void
bsrilu02_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_size, bsrilu02_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSbsrilu02_bufferSizeExt(h, dir_a, mb, nnzb,
		descr_a, values, starts, indices, block_size, info, buffer_size));
}

inline void
bsrilu02_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_size, bsrilu02_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDbsrilu02_bufferSizeExt(h, dir_a, mb, nnzb,
		descr_a, values, starts, indices, block_size, info, buffer_size));
}

inline void
bsrilu02_analysis(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsrilu02_analysis(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, policy, buffer));
}

inline void
bsrilu02_analysis(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsrilu02_analysis(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, policy, buffer));
}

inline void
bsrilu02(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsrilu02(h, dir_a, mb, nnzb, descr_a, values,
		starts, indices, block_dim, info, policy, buffer));
}

inline void
bsrilu02(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_dim, bsrilu02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsrilu02(h, dir_a, mb, nnzb, descr_a, values,
		starts, indices, block_dim, info, policy, buffer));
}

} // namespace cusparse
