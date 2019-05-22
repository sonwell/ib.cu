#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

inline void
getrf_buffer_size(dense::handle_t h, int m, int n, float* a, int lda,
		int* work_size)
{
	throw_if_error(cusolverDnSgetrf_bufferSize(h, m, n, a, lda, work_size));
}

inline void
getrf_buffer_size(dense::handle_t h, int m, int n, double* a, int lda,
		int* work_size)
{
	throw_if_error(cusolverDnDgetrf_bufferSize(h, m, n, a, lda, work_size));
}

inline void
getrf(dense::handle_t h, int m, int n, float* a, int lda, float* workspace,
		int* pivots, int* info)
{
	throw_if_error(cusolverDnSgetrf(h, m, n, a, lda, workspace, pivots,
		info));
}

inline void
getrf(dense::handle_t h, int m, int n, double* a, int lda, double* workspace,
		int* pivots, int* info)
{
	throw_if_error(cusolverDnDgetrf(h, m, n, a, lda, workspace, pivots,
		info));
}

} // namespace cusolver
