#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"
#include "diagonal_type.h"
#include "fill_mode.h"

namespace cublas {

inline void
tpsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, const float* a, float* x, int incx)
{
	throw_if_error(cublasStpsv(h, uplo, trans, diag, n, a, x, incx));
}

inline void
tpsv(handle_t h, fill_mode_adaptor uplo, operation_adaptor trans,
		diagonal_type_adaptor diag, int n, const double* a, double* x, int incx)
{
	throw_if_error(cublasDtpsv(h, uplo, trans, diag, n, a, x, incx));
}

} // namespace cublas
