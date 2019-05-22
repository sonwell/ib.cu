#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
gthrz(handle& h, int nnz, float* y, float* values, const int* indices,
		index_base_adaptor base)
{
	throw_if_error(cusparseSgthrz(h, nnz, y, values, indices, base));
}

inline void
gthrz(handle& h, int nnz, double* y, double* values, const int* indices,
		index_base_adaptor base)
{
	throw_if_error(cusparseDgthrz(h, nnz, y, values, indices, base));
}

} // namespace cusparse