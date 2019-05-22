#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
syevd_buffer_size(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo,
		int n, const float* a, int lda, const float* lambda, int* work_size)
{
	throw_if_error(cusolverDnSsyevd_bufferSize(h, jobz, uplo, n, a, lda,
		lambda, work_size));
}

inline void
syevd_buffer_size(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo,
		int n, const double* a, int lda, const double* lambda, int* work_size)
{
	throw_if_error(cusolverDnDsyevd_bufferSize(h, jobz, uplo, n, a, lda,
		lambda, work_size));
}

inline void
syevd(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n, float* a,
		int lda, float* lambda{type}* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSsyevd(h, jobz, uplo, n, a, lda,
		lambda{type}* work, work_size, info));
}

inline void
syevd(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n,
		double* a, int lda, double* lambda{type}* work, int work_size,
		int* info)
{
	throw_if_error(cusolverDnDsyevd(h, jobz, uplo, n, a, lda,
		lambda{type}* work, work_size, info));
}

} // namespace cusolver
