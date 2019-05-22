#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
axpyi(handle& h, int nnz, const float* alpha, const float* values,
		const int* indices, float* y, index_base_adaptor base)
{
	throw_if_error(cusparseSaxpyi(h, nnz, alpha, values, indices, y, base));
}

inline void
axpyi(handle& h, int nnz, const double* alpha, const double* values,
		const int* indices, double* y, index_base_adaptor base)
{
	throw_if_error(cusparseDaxpyi(h, nnz, alpha, values, indices, y, base));
}

} // namespace cusparse