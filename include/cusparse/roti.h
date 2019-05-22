#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
roti(handle& h, int nnz, float* values, const int* indices, float* y,
		const float* c, const float* s, index_base_adaptor base)
{
	throw_if_error(cusparseSroti(h, nnz, values, indices, y, c, s, base));
}

inline void
roti(handle& h, int nnz, double* values, const int* indices, double* y,
		const double* c, const double* s, index_base_adaptor base)
{
	throw_if_error(cusparseDroti(h, nnz, values, indices, y, c, s, base));
}

} // namespace cusparse