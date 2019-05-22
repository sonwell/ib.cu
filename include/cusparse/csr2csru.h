#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "csru2csr_info.h"

namespace cusparse {

inline void
csr2csru(handle& h, int m, int n, int nnz, const matrix_description descr_a,
		float* values, const int* starts, int* indices, csru2csr_info info,
		void* buffer)
{
	throw_if_error(cusparseScsr2csru(h, m, n, nnz, descr_a, values, starts,
		indices, info, buffer));
}

inline void
csr2csru(handle& h, int m, int n, int nnz, const matrix_description descr_a,
		double* values, const int* starts, int* indices, csru2csr_info info,
		void* buffer)
{
	throw_if_error(cusparseDcsr2csru(h, m, n, nnz, descr_a, values, starts,
		indices, info, buffer));
}

} // namespace cusparse
