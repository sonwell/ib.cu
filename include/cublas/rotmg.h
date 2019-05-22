#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
rotmg(handle_t h, float* d1, float* d2, float* x1, const float* y1, float* param)
{
	throw_if_error(cublasSrotmg(h, d1, d2, x1, y1, param));
}

inline void
rotmg(handle_t h, double* d1, double* d2, double* x1, const double* y1,
		double* param)
{
	throw_if_error(cublasDrotmg(h, d1, d2, x1, y1, param));
}

} // namespace cublas
