#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "diagonal_type.h"
#include "side_mode.h"
#include "fill_mode.h"

namespace cublas {

inline void
trmm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const float* alpha, const float* a, int lda, const float* b, int ldb,
		float* c, int ldc)
{
	throw_if_error(cublasStrmm(h, side, uplo, trans, diag, m, n, alpha, a,
		lda, b, ldb, c, ldc));
}

inline void
trmm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const double* alpha, const double* a, int lda, const double* b, int ldb,
		double* c, int ldc)
{
	throw_if_error(cublasDtrmm(h, side, uplo, trans, diag, m, n, alpha, a,
		lda, b, ldb, c, ldc));
}

} // namespace cublas
