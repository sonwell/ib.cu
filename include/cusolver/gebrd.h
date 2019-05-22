#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

inline void
gebrd_buffer_size(dense::handle_t h, int m, int n, int* work_size)
{
	throw_if_error(cusolverDnSgebrd_bufferSize(h, m, n, work_size));
}

inline void
gebrd_buffer_size(dense::handle_t h, int m, int n, int* work_size)
{
	throw_if_error(cusolverDnDgebrd_bufferSize(h, m, n, work_size));
}

inline void
gebrd(dense::handle_t h, int m, int n, float* a, int lda, float* d, float* e,
		float* tau_q, float* tau_p, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSgebrd(h, m, n, a, lda, d, e, tau_q, tau_p,
		work, work_size, info));
}

inline void
gebrd(dense::handle_t h, int m, int n, double* a, int lda, double* d, double* e,
		double* tau_q, double* tau_p, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDgebrd(h, m, n, a, lda, d, e, tau_q, tau_p,
		work, work_size, info));
}

} // namespace cusolver
