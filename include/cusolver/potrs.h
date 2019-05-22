#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
potrs(dense::handle_t h, fill_mode_adaptor uplo, int n, int nrhs, const float* a,
		int lda, float* b, int ldb, int* info)
{
	throw_if_error(cusolverDnSpotrs(h, uplo, n, nrhs, a, lda, b, ldb, info));
}

inline void
potrs(dense::handle_t h, fill_mode_adaptor uplo, int n, int nrhs, const double* a,
		int lda, double* b, int ldb, int* info)
{
	throw_if_error(cusolverDnDpotrs(h, uplo, n, nrhs, a, lda, b, ldb, info));
}

inline void
potrs_batched(dense::handle_t h, fill_mode_adaptor uplo, int n, int nrhs,
		float* a[], int lda, float* b[], int ldb, int* info, int batch_size)
{
	throw_if_error(cusolverDnSpotrsBatched(h, uplo, n, nrhs, a[], lda, b[],
		ldb, info, batch_size));
}

inline void
potrs_batched(dense::handle_t h, fill_mode_adaptor uplo, int n, int nrhs,
		double* a[], int lda, double* b[], int ldb, int* info, int batch_size)
{
	throw_if_error(cusolverDnDpotrsBatched(h, uplo, n, nrhs, a[], lda, b[],
		ldb, info, batch_size));
}

} // namespace cusolver
