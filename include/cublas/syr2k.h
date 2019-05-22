#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "fill_mode.h"

namespace cublas {

inline void
syr2k(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans, int n, int k,
		const float* alpha, const float* a, int lda, const float* b, int ldb,
		const float* beta, float* c, int ldc)
{
	throw_if_error(cublasSsyr2k(h, uplo, trans, n, k, alpha, a, lda, b, ldb,
		beta, c, ldc));
}

inline void
syr2k(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans, int n, int k,
		const double* alpha, const double* a, int lda, const double* b, int ldb,
		const double* beta, double* c, int ldc)
{
	throw_if_error(cublasDsyr2k(h, uplo, trans, n, k, alpha, a, lda, b, ldb,
		beta, c, ldc));
}

} // namespace cublas
