#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "diagonal_type.h"
#include "side_mode.h"
#include "operation.h"
#include "fill_mode.h"

namespace cublas {

inline void
trsm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const float* alpha, const float* a, int lda, float* b, int ldb)
{
	throw_if_error(cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, a,
		lda, b, ldb));
}

inline void
trsm(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const double* alpha, const double* a, int lda, double* b, int ldb)
{
	throw_if_error(cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, a,
		lda, b, ldb));
}

inline void
trsm_batched(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const float* alpha, const float* const a[], int lda, float* const b[],
		int ldb, int batch_count)
{
	throw_if_error(cublasStrsmBatched(h, side, uplo, trans, diag, m, n,
		alpha, a[], lda, b[], ldb, batch_count));
}

inline void
trsm_batched(handle_t h, side_mode_adaptor side, fill_mode_adaptor uplo,
		operation_adaptor trans, diagonal_type_adaptor diag, int m, int n,
		const double* alpha, const double* const a[], int lda,
		double* const b[], int ldb, int batch_count)
{
	throw_if_error(cublasDtrsmBatched(h, side, uplo, trans, diag, m, n,
		alpha, a[], lda, b[], ldb, batch_count));
}

} // namespace cublas
