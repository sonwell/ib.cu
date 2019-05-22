#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
sctr(handle& h, int nnz, const float* values, const int* indices, float* y,
		index_base_adaptor base)
{
	throw_if_error(cusparseSsctr(h, nnz, values, indices, y, base));
}

inline void
sctr(handle& h, int nnz, const double* values, const int* indices, double* y,
		index_base_adaptor base)
{
	throw_if_error(cusparseDsctr(h, nnz, values, indices, y, base));
}

} // namespace cusparse