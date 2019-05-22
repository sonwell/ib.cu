#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
tpttr(handle_t h, fill_mode_adaptor uplo, int n, const float* a, float* b,
		int ldb)
{
	throw_if_error(cublasStpttr(h, uplo, n, a, b, ldb));
}

inline void
tpttr(handle_t h, fill_mode_adaptor uplo, int n, const double* a, double* b,
		int ldb)
{
	throw_if_error(cublasDtpttr(h, uplo, n, a, b, ldb));
}

} // namespace cublas
