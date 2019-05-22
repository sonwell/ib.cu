#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "side_mode.h"
#include "fill_mode.h"

namespace cublas {

inline void
symm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo, int m, int n,
		const float* alpha, const float* a, int lda, const float* b, int ldb,
		const float* beta, float* c, int ldc)
{
	throw_if_error(cublasSsymm(h, side, uplo, m, n, alpha, a, lda, b, ldb,
		beta, c, ldc));
}

inline void
symm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo, int m, int n,
		const double* alpha, const double* a, int lda, const double* b, int ldb,
		const double* beta, double* c, int ldc)
{
	throw_if_error(cublasDsymm(h, side, uplo, m, n, alpha, a, lda, b, ldb,
		beta, c, ldc));
}

} // namespace cublas
