#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

inline void
geqrf_buffer_size(dense::handle_t h, int m, int n, float* a, int lda,
		int* work_size)
{
	throw_if_error(cusolverDnSgeqrf_bufferSize(h, m, n, a, lda, work_size));
}

inline void
geqrf_buffer_size(dense::handle_t h, int m, int n, double* a, int lda,
		int* work_size)
{
	throw_if_error(cusolverDnDgeqrf_bufferSize(h, m, n, a, lda, work_size));
}

inline void
geqrf(dense::handle_t h, int m, int n, float* a, int lda, float* tau,
		float* workspace, int work_size, int* info)
{
	throw_if_error(cusolverDnSgeqrf(h, m, n, a, lda, tau, workspace,
		work_size, info));
}

inline void
geqrf(dense::handle_t h, int m, int n, double* a, int lda, double* tau,
		double* workspace, int work_size, int* info)
{
	throw_if_error(cusolverDnDgeqrf(h, m, n, a, lda, tau, workspace,
		work_size, info));
}

} // namespace cusolver
