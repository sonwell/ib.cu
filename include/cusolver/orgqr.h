#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

inline void
orgqr_buffer_size(dense::handle_t h, int m, int n, int k, const float* a, int lda,
		const float* tau, int* work_size)
{
	throw_if_error(cusolverDnSorgqr_bufferSize(h, m, n, k, a, lda, tau,
		work_size));
}

inline void
orgqr_buffer_size(dense::handle_t h, int m, int n, int k, const double* a,
		int lda, const double* tau, int* work_size)
{
	throw_if_error(cusolverDnDorgqr_bufferSize(h, m, n, k, a, lda, tau,
		work_size));
}

inline void
orgqr(dense::handle_t h, int m, int n, int k, float* a, int lda, const float* tau,
		float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSorgqr(h, m, n, k, a, lda, tau, work,
		work_size, info));
}

inline void
orgqr(dense::handle_t h, int m, int n, int k, double* a, int lda,
		const double* tau, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDorgqr(h, m, n, k, a, lda, tau, work,
		work_size, info));
}

} // namespace cusolver
