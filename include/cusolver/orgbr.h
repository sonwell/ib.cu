#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "side_mode.h"

namespace cusolver {

inline void
orgbr_buffer_size(dense::handle_t h, side_mode_adaptor side, int m, int n, int k,
		const float* a, int lda, const float* tau, int* work_size)
{
	throw_if_error(cusolverDnSorgbr_bufferSize(h, side, m, n, k, a, lda,
		tau, work_size));
}

inline void
orgbr_buffer_size(dense::handle_t h, side_mode_adaptor side, int m, int n, int k,
		const double* a, int lda, const double* tau, int* work_size)
{
	throw_if_error(cusolverDnDorgbr_bufferSize(h, side, m, n, k, a, lda,
		tau, work_size));
}

inline void
orgbr(dense::handle_t h, side_mode_adaptor side, int m, int n, int k, float* a,
		int lda, const float* tau, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSorgbr(h, side, m, n, k, a, lda, tau, work,
		work_size, info));
}

inline void
orgbr(dense::handle_t h, side_mode_adaptor side, int m, int n, int k, double* a,
		int lda, const double* tau, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDorgbr(h, side, m, n, k, a, lda, tau, work,
		work_size, info));
}

} // namespace cusolver
