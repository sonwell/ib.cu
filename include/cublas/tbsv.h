#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "diagonal_type.h"
#include "fill_mode.h"

namespace cublas {

inline void
tbsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, int k, const float* a, int lda,
		float* x, int incx)
{
	throw_if_error(cublasStbsv(h, uplo, trans, diag, n, k, a, lda, x, incx));
}

inline void
tbsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, int k, const double* a, int lda,
		double* x, int incx)
{
	throw_if_error(cublasDtbsv(h, uplo, trans, diag, n, k, a, lda, x, incx));
}

} // namespace cublas
