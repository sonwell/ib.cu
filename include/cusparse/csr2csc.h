#pragma once
#include "types.h"
#include "handle.h"
#include "action.h"
#include "index_base.h"

namespace cusparse {

inline void
csr2csc(handle& h, int m, int n, int nnz, const float* values,
		const int* starts, const int* indices, float* values, int* starts,
		int* indices, action_adaptor copy, index_base_adaptor base)
{
	throw_if_error(cusparseScsr2csc(h, m, n, nnz, values, starts, indices,
		values, starts, indices, copy, base));
}

inline void
csr2csc(handle& h, int m, int n, int nnz, const double* values,
		const int* starts, const int* indices, double* values, int* starts,
		int* indices, action_adaptor copy, index_base_adaptor base)
{
	throw_if_error(cusparseDcsr2csc(h, m, n, nnz, values, starts, indices,
		values, starts, indices, copy, base));
}

} // namespace cusparse