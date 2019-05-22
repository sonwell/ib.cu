#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
gebsr2gebsr_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nb,
		int nnzb, const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, int row_block_dim_c, int col_block_dim_c,
		int* buffer_size)
{
	throw_if_error(cusparseSgebsr2gebsr_bufferSize(h, dir_a, mb, nb, nnzb,
		descr_a, values_a, starts_a, indices_a, row_block_dim_a,
		col_block_dim_a, row_block_dim_c, col_block_dim_c, buffer_size));
}

inline void
gebsr2gebsr_buffer_size(handle& h, direction_adaptor dir_a, int mb, int nb,
		int nnzb, const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, int row_block_dim_c, int col_block_dim_c,
		int* buffer_size)
{
	throw_if_error(cusparseDgebsr2gebsr_bufferSize(h, dir_a, mb, nb, nnzb,
		descr_a, values_a, starts_a, indices_a, row_block_dim_a,
		col_block_dim_a, row_block_dim_c, col_block_dim_c, buffer_size));
}

inline void
gebsr2gebsr_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nb,
		int nnzb, const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, int row_block_dim_c, int col_block_dim_c,
		size_t* buffer_size)
{
	throw_if_error(cusparseSgebsr2gebsr_bufferSizeExt(h, dir_a, mb, nb,
		nnzb, descr_a, values_a, starts_a, indices_a, row_block_dim_a,
		col_block_dim_a, row_block_dim_c, col_block_dim_c, buffer_size));
}

inline void
gebsr2gebsr_buffer_size_ext(handle& h, direction_adaptor dir_a, int mb, int nb,
		int nnzb, const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, int row_block_dim_c, int col_block_dim_c,
		size_t* buffer_size)
{
	throw_if_error(cusparseDgebsr2gebsr_bufferSizeExt(h, dir_a, mb, nb,
		nnzb, descr_a, values_a, starts_a, indices_a, row_block_dim_a,
		col_block_dim_a, row_block_dim_c, col_block_dim_c, buffer_size));
}

inline void
gebsr2gebsr_nnz(handle& h, direction_adaptor dir_a, int mb, int nb, int nnzb,
		const matrix_description descr_a, const int* starts_a,
		const int* indices_a, int row_block_dim_a, int col_block_dim_a,
		const matrix_description descr_c, int* starts_c, int row_block_dim_c,
		int col_block_dim_c, int* nnz_total, void* buffer)
{
	throw_if_error(cusparseXgebsr2gebsrNnz(h, dir_a, mb, nb, nnzb, descr_a,
		starts_a, indices_a, row_block_dim_a, col_block_dim_a, descr_c,
		starts_c, row_block_dim_c, col_block_dim_c, nnz_total, buffer));
}

inline void
gebsr2gebsr(handle& h, direction_adaptor dir_a, int mb, int nb, int nnzb,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, const matrix_description descr_c, float* values_c,
		int* starts_c, int* indices_c, int row_block_dim_c, int col_block_dim_c,
		void* buffer)
{
	throw_if_error(cusparseSgebsr2gebsr(h, dir_a, mb, nb, nnzb, descr_a,
		values_a, starts_a, indices_a, row_block_dim_a, col_block_dim_a,
		descr_c, values_c, starts_c, indices_c, row_block_dim_c,
		col_block_dim_c, buffer));
}

inline void
gebsr2gebsr(handle& h, direction_adaptor dir_a, int mb, int nb, int nnzb,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim_a,
		int col_block_dim_a, const matrix_description descr_c, double* values_c,
		int* starts_c, int* indices_c, int row_block_dim_c, int col_block_dim_c,
		void* buffer)
{
	throw_if_error(cusparseDgebsr2gebsr(h, dir_a, mb, nb, nnzb, descr_a,
		values_a, starts_a, indices_a, row_block_dim_a, col_block_dim_a,
		descr_c, values_c, starts_c, indices_c, row_block_dim_c,
		col_block_dim_c, buffer));
}

} // namespace cusparse
