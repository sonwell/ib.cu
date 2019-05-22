#pragma once
#include "types.h"
#include "handle.h"
#include "action.h"
#include "index_base.h"

namespace cusparse {

inline void
gebsr2gebsc_buffer_size(handle& h, int mb, int nb, int nnzb,
		const float* values, const int* starts, const int* indices,
		int row_block_dim, int col_block_dim, int* buffer_size)
{
	throw_if_error(cusparseSgebsr2gebsc_bufferSize(h, mb, nb, nnzb, values,
		starts, indices, row_block_dim, col_block_dim, buffer_size));
}

inline void
gebsr2gebsc_buffer_size(handle& h, int mb, int nb, int nnzb,
		const double* values, const int* starts, const int* indices,
		int row_block_dim, int col_block_dim, int* buffer_size)
{
	throw_if_error(cusparseDgebsr2gebsc_bufferSize(h, mb, nb, nnzb, values,
		starts, indices, row_block_dim, col_block_dim, buffer_size));
}

inline void
gebsr2gebsc_buffer_size_ext(handle& h, int mb, int nb, int nnzb,
		const float* values, const int* starts, const int* indices,
		int row_block_dim, int col_block_dim, size_t* buffer_size)
{
	throw_if_error(cusparseSgebsr2gebsc_bufferSizeExt(h, mb, nb, nnzb,
		values, starts, indices, row_block_dim, col_block_dim, buffer_size));
}

inline void
gebsr2gebsc_buffer_size_ext(handle& h, int mb, int nb, int nnzb,
		const double* values, const int* starts, const int* indices,
		int row_block_dim, int col_block_dim, size_t* buffer_size)
{
	throw_if_error(cusparseDgebsr2gebsc_bufferSizeExt(h, mb, nb, nnzb,
		values, starts, indices, row_block_dim, col_block_dim, buffer_size));
}

inline void
gebsr2gebsc(handle& h, int mb, int nb, int nnzb, const float* values,
		const int* starts, const int* indices, int row_block_dim,
		int col_block_dim, float* values, int* indices, int* starts,
		action_adaptor copy, index_base_adaptor base, void* buffer)
{
	throw_if_error(cusparseSgebsr2gebsc(h, mb, nb, nnzb, values, starts,
		indices, row_block_dim, col_block_dim, values, indices, starts,
		copy, base, buffer));
}

inline void
gebsr2gebsc(handle& h, int mb, int nb, int nnzb, const double* values,
		const int* starts, const int* indices, int row_block_dim,
		int col_block_dim, double* values, int* indices, int* starts,
		action_adaptor copy, index_base_adaptor base, void* buffer)
{
	throw_if_error(cusparseDgebsr2gebsc(h, mb, nb, nnzb, values, starts,
		indices, row_block_dim, col_block_dim, values, indices, starts,
		copy, base, buffer));
}

} // namespace cusparse