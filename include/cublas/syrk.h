#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "fill_mode.h"

namespace cublas {

inline void
syrk(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans, int n, int k,
		const float* alpha, const float* a, int lda, const float* beta,
		float* c, int ldc)
{
	throw_if_error(cublasSsyrk(h, uplo, trans, n, k, alpha, a, lda, beta, c,
		ldc));
}

inline void
syrk(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans, int n, int k,
		const double* alpha, const double* a, int lda, const double* beta,
		double* c, int ldc)
{
	throw_if_error(cublasDsyrk(h, uplo, trans, n, k, alpha, a, lda, beta, c,
		ldc));
}

} // namespace cublas
