#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
csr2gebsr_buffer_size(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, int* buffer_size)
{
	throw_if_error(cusparseScsr2gebsr_bufferSize(h, dir_a, m, n, descr_a,
		values_a, starts_a, indices_a, row_block_dim, col_block_dim,
		buffer_size));
}

inline void
csr2gebsr_buffer_size(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, int* buffer_size)
{
	throw_if_error(cusparseDcsr2gebsr_bufferSize(h, dir_a, m, n, descr_a,
		values_a, starts_a, indices_a, row_block_dim, col_block_dim,
		buffer_size));
}

inline void
csr2gebsr_buffer_size_ext(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, size_t* buffer_size)
{
	throw_if_error(cusparseScsr2gebsr_bufferSizeExt(h, dir_a, m, n, descr_a,
		values_a, starts_a, indices_a, row_block_dim, col_block_dim,
		buffer_size));
}

inline void
csr2gebsr_buffer_size_ext(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int row_block_dim,
		int col_block_dim, size_t* buffer_size)
{
	throw_if_error(cusparseDcsr2gebsr_bufferSizeExt(h, dir_a, m, n, descr_a,
		values_a, starts_a, indices_a, row_block_dim, col_block_dim,
		buffer_size));
}

inline void
csr2gebsr_nnz(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const int* starts_a,
		const int* indices_a, const matrix_description descr_c, int* starts_c,
		int row_block_dim, int col_block_dim, int* nnz_total, void* buffer)
{
	throw_if_error(cusparseXcsr2gebsrNnz(h, dir_a, m, n, descr_a, starts_a,
		indices_a, descr_c, starts_c, row_block_dim, col_block_dim,
		nnz_total, buffer));
}

inline void
csr2gebsr(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a,
		const matrix_description descr_c, float* values_c, int* starts_c,
		int* indices_c, int row_block_dim, int col_block_dim, void* buffer)
{
	throw_if_error(cusparseScsr2gebsr(h, dir_a, m, n, descr_a, values_a,
		starts_a, indices_a, descr_c, values_c, starts_c, indices_c,
		row_block_dim, col_block_dim, buffer));
}

inline void
csr2gebsr(handle& h, direction_adaptor dir_a, int m, int n,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a,
		const matrix_description descr_c, double* values_c, int* starts_c,
		int* indices_c, int row_block_dim, int col_block_dim, void* buffer)
{
	throw_if_error(cusparseDcsr2gebsr(h, dir_a, m, n, descr_a, values_a,
		starts_a, indices_a, descr_c, values_c, starts_c, indices_c,
		row_block_dim, col_block_dim, buffer));
}

} // namespace cusparse
