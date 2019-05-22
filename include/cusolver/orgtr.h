#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
orgtr_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n,
		const float* a, int lda, const float* tau, int* work_size)
{
	throw_if_error(cusolverDnSorgtr_bufferSize(h, uplo, n, a, lda, tau,
		work_size));
}

inline void
orgtr_buffer_size(dense::handle_t h, fill_mode_adaptor uplo, int n,
		const double* a, int lda, const double* tau, int* work_size)
{
	throw_if_error(cusolverDnDorgtr_bufferSize(h, uplo, n, a, lda, tau,
		work_size));
}

inline void
orgtr(dense::handle_t h, fill_mode_adaptor uplo, int n, float* a, int lda,
		const float* tau, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSorgtr(h, uplo, n, a, lda, tau, work,
		work_size, info));
}

inline void
orgtr(dense::handle_t h, fill_mode_adaptor uplo, int n, double* a, int lda,
		const double* tau, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDorgtr(h, uplo, n, a, lda, tau, work,
		work_size, info));
}

} // namespace cusolver
