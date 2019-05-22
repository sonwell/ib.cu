#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
sytrd_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n,
		const float* a, int lda, const float* d, const float* e,
		const float* tau, int* work_size)
{
	throw_if_error(cusolverDnSsytrd_bufferSize(h, uplo, n, a, lda, d, e,
		tau, work_size));
}

inline void
sytrd_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n,
		const double* a, int lda, const double* d, const double* e,
		const double* tau, int* work_size)
{
	throw_if_error(cusolverDnDsytrd_bufferSize(h, uplo, n, a, lda, d, e,
		tau, work_size));
}

inline void
sytrd(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a, int lda,
		float* d, float* e, float* tau, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSsytrd(h, uplo, n, a, lda, d, e, tau, work,
		work_size, info));
}

inline void
sytrd(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a, int lda,
		double* d, double* e, double* tau, double* work, int work_size,
		int* info)
{
	throw_if_error(cusolverDnDsytrd(h, uplo, n, a, lda, d, e, tau, work,
		work_size, info));
}

} // namespace cusolver
