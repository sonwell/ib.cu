#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
potrf_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a,
		int lda, int* work_size)
{
	throw_if_error(cusolverDnSpotrf_bufferSize(h, uplo, n, a, lda,
		work_size));
}

inline void
potrf_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a,
		int lda, int* work_size)
{
	throw_if_error(cusolverDnDpotrf_bufferSize(h, uplo, n, a, lda,
		work_size));
}

inline void
potrf(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a, int lda,
		float* workspace, int work_size, int* info)
{
	throw_if_error(cusolverDnSpotrf(h, uplo, n, a, lda, workspace,
		work_size, info));
}

inline void
potrf(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a, int lda,
		double* workspace, int work_size, int* info)
{
	throw_if_error(cusolverDnDpotrf(h, uplo, n, a, lda, workspace,
		work_size, info));
}

inline void
potrf_batched(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a[],
		int lda, int* info_array, int batch_size)
{
	throw_if_error(cusolverDnSpotrfBatched(h, uplo, n, a[], lda, info_array,
		batch_size));
}

inline void
potrf_batched(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a[],
		int lda, int* info_array, int batch_size)
{
	throw_if_error(cusolverDnDpotrfBatched(h, uplo, n, a[], lda, info_array,
		batch_size));
}

} // namespace cusolver
