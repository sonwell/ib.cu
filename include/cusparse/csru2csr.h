#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "csru2csr_info.h"

namespace cusparse {

inline void
csru2csr_buffer_size_ext(handle& h, int m, int n, int nnz, float* values,
		const int* starts, int* indices, csru2csr_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseScsru2csr_bufferSizeExt(h, m, n, nnz, values,
		starts, indices, info, buffer_size));
}

inline void
csru2csr_buffer_size_ext(handle& h, int m, int n, int nnz, double* values,
		const int* starts, int* indices, csru2csr_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDcsru2csr_bufferSizeExt(h, m, n, nnz, values,
		starts, indices, info, buffer_size));
}

inline void
csru2csr(handle& h, int m, int n, int nnz, const matrix_description descr_a,
		float* values, const int* starts, int* indices, csru2csr_info info,
		void* buffer)
{
	throw_if_error(cusparseScsru2csr(h, m, n, nnz, descr_a, values, starts,
		indices, info, buffer));
}

inline void
csru2csr(handle& h, int m, int n, int nnz, const matrix_description descr_a,
		double* values, const int* starts, int* indices, csru2csr_info info,
		void* buffer)
{
	throw_if_error(cusparseDcsru2csr(h, m, n, nnz, descr_a, values, starts,
		indices, info, buffer));
}

} // namespace cusparse
