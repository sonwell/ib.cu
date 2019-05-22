#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "diagonal_type.h"
#include "fill_mode.h"

namespace cublas {

inline void
trsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, const float* a, int lda, float* x,
		int incx)
{
	throw_if_error(cublasStrsv(h, uplo, trans, diag, n, a, lda, x, incx));
}

inline void
trsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, const double* a, int lda, double* x,
		int incx)
{
	throw_if_error(cublasDtrsv(h, uplo, trans, diag, n, a, lda, x, incx));
}

} // namespace cublas
