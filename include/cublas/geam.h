#pragma once
#include "types.h"
#include "handle.h"
#include "operation.h"
#include "exceptions.h"

namespace cublas {

inline void
geam(handle_t h, operation_adaptor transa, operation_adaptor transb, int m, int n,
		const float* alpha, const float* a, int lda, const float* beta,
		const float* b, int ldb, float* c, int ldc)
{
	throw_if_error(cublasSgeam(h, transa, transb, m, n, alpha, a, lda, beta,
		b, ldb, c, ldc));
}

inline void
geam(handle_t h, operation_adaptor transa, operation_adaptor transb, int m, int n,
		const double* alpha, const double* a, int lda, const double* beta,
		const double* b, int ldb, double* c, int ldc)
{
	throw_if_error(cublasDgeam(h, transa, transb, m, n, alpha, a, lda, beta,
		b, ldb, c, ldc));
}

} // namespace cublas
