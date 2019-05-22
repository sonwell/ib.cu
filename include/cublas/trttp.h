#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cublas {

inline void
trttp(handle_t h, fill_mode_adaptor uplo, int n, const float* a, int lda,
		float* b)
{
	throw_if_error(cublasStrttp(h, uplo, n, a, lda, b));
}

inline void
trttp(handle_t h, fill_mode_adaptor uplo, int n, const double* a, int lda,
		double* b)
{
	throw_if_error(cublasDtrttp(h, uplo, n, a, lda, b));
}

} // namespace cublas
