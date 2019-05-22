#pragma once
#include "types.h"
#include "handle.h"
#include "permutation.h"

namespace cusparse {

inline void
coosort_buffer_size_ext(handle& h, int m, int n, int nnz, const int* rows_a,
		const int* cols_a, size_t* buffer_size)
{
	throw_if_error(cusparseXcoosort_bufferSizeExt(h, m, n, nnz, rows_a,
		cols_a, buffer_size));
}

inline void
coosort_by_row(handle& h, int m, int n, int nnz, int* rows_a, int* cols_a,
		int* permutation, void* buffer)
{
	throw_if_error(cusparseXcoosortByRow(h, m, n, nnz, rows_a, cols_a,
		permutation, buffer));
}

inline void
coosort_by_column(handle& h, int m, int n, int nnz, int* rows_a, int* cols_a,
		int* permutation, void* buffer)
{
	throw_if_error(cusparseXcoosortByColumn(h, m, n, nnz, rows_a, cols_a,
		permutation, buffer));
}

} // namespace cusparse
