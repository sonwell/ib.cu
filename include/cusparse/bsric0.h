#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "solve_policy.h"
#include "direction.h"
#include "bsric02_info.h"

namespace cusparse {

inline void
bsric02_zero_pivot(handle& h, bsric02_info info, int* position)
{
	throw_if_error(cusparseXbsric02_zeroPivot(h, info, position));
}

inline void
bsric02_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_dim, bsric02_info info, int* buffer_size)
{
	throw_if_error(cusparseSbsric02_bufferSize(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, buffer_size));
}

inline void
bsric02_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_dim, bsric02_info info, int* buffer_size)
{
	throw_if_error(cusparseDbsric02_bufferSize(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, buffer_size));
}

inline void
bsric02_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_size, bsric02_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSbsric02_bufferSizeExt(h, dir_a, mb, nnzb,
		descr_a, values, starts, indices, block_size, info, buffer_size));
}

inline void
bsric02_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_size, bsric02_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDbsric02_bufferSizeExt(h, dir_a, mb, nnzb,
		descr_a, values, starts, indices, block_size, info, buffer_size));
}

inline void
bsric02_analysis(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, const float* values,
		const int* starts, const int* indices, int block_dim, bsric02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsric02_analysis(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, policy, buffer));
}

inline void
bsric02_analysis(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, const double* values,
		const int* starts, const int* indices, int block_dim, bsric02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsric02_analysis(h, dir_a, mb, nnzb, descr_a,
		values, starts, indices, block_dim, info, policy, buffer));
}

inline void
bsric02(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, int block_dim, bsric02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseSbsric02(h, dir_a, mb, nnzb, descr_a, values,
		starts, indices, block_dim, info, policy, buffer));
}

inline void
bsric02(handle& h, direction_adaptor dir_a, int mb, int nnzb,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, int block_dim, bsric02_info info,
		solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDbsric02(h, dir_a, mb, nnzb, descr_a, values,
		starts, indices, block_dim, info, policy, buffer));
}

} // namespace cusparse
