#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
doti(handle& h, int nnz, const float* values, const int* indices,
		const float* y, float* result, index_base_adaptor base)
{
	throw_if_error(cusparseSdoti(h, nnz, values, indices, y, result, base));
}

inline void
doti(handle& h, int nnz, const double* values, const int* indices,
		const double* y, double* result, index_base_adaptor base)
{
	throw_if_error(cusparseDdoti(h, nnz, values, indices, y, result, base));
}

} // namespace cusparse