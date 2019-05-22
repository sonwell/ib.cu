#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

inline void
rotg(handle_t h, float* a, float* b, float* c, float* s)
{
	throw_if_error(cublasSrotg(h, a, b, c, s));
}

inline void
rotg(handle_t h, double* a, double* b, double* c, double* s)
{
	throw_if_error(cublasDrotg(h, a, b, c, s));
}

} // namespace cublas
