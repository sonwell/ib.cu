#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "permutation.h"

namespace cusparse {

inline void
cscsort_buffer_size_ext(handle& h, int m, int n, int nnz, const int* starts_a,
		const int* indices_a, size_t* buffer_size)
{
	throw_if_error(cusparseXcscsort_bufferSizeExt(h, m, n, nnz, starts_a,
		indices_a, buffer_size));
}

inline void
cscsort(handle& h, int m, int n, int nnz, const matrix_description descr_a,
		const int* starts_a, int* indices_a, int* permutation, void* buffer)
{
	throw_if_error(cusparseXcscsort(h, m, n, nnz, descr_a, starts_a,
		indices_a, permutation, buffer));
}

} // namespace cusparse
