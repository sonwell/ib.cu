#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
sytrf_buffer_size(dense::handle_t h, int n, float* a, int lda, int* work_size)
{
	throw_if_error(cusolverDnSsytrf_bufferSize(h, n, a, lda, work_size));
}

inline void
sytrf_buffer_size(dense::handle_t h, int n, double* a, int lda, int* work_size)
{
	throw_if_error(cusolverDnDsytrf_bufferSize(h, n, a, lda, work_size));
}

inline void
sytrf(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a, int lda,
		int* pivots, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSsytrf(h, uplo, n, a, lda, pivots, work,
		work_size, info));
}

inline void
sytrf(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a, int lda,
		int* pivots, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDsytrf(h, uplo, n, a, lda, pivots, work,
		work_size, info));
}

} // namespace cusolver
