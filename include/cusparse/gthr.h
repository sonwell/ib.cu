#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
gthr(handle& h, int nnz, const float* y, float* values, const int* indices,
		index_base_adaptor base)
{
	throw_if_error(cusparseSgthr(h, nnz, y, values, indices, base));
}

inline void
gthr(handle& h, int nnz, const double* y, double* values, const int* indices,
		index_base_adaptor base)
{
	throw_if_error(cusparseDgthr(h, nnz, y, values, indices, base));
}

} // namespace cusparse