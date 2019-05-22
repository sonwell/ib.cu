#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"
#include "syevj_info.h"

namespace cusolver {

inline void
syevj_batched_buffer_size(dense::handle_t h, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, const float* a, int lda,
		const float* lambda, int* work_size, syevj_info params, int batch_size)
{
	throw_if_error(cusolverDnSsyevjBatched_bufferSize(h, jobz, uplo, n, a,
		lda, lambda, work_size, params, batch_size));
}

inline void
syevj_batched_buffer_size(dense::handle_t h, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, const double* a, int lda,
		const double* lambda, int* work_size, syevj_info params, int batch_size)
{
	throw_if_error(cusolverDnDsyevjBatched_bufferSize(h, jobz, uplo, n, a,
		lda, lambda, work_size, params, batch_size));
}

inline void
syevj_batched(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n,
		float* a, int lda, float* lambda, float* work, int work_size, int* info,
		syevj_info params, int batch_size)
{
	throw_if_error(cusolverDnSsyevjBatched(h, jobz, uplo, n, a, lda,
		lambda, work, work_size, info, params, batch_size));
}

inline void
syevj_batched(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n,
		double* a, int lda, double* lambda, double* work, int work_size,
		int* info, syevj_info params, int batch_size)
{
	throw_if_error(cusolverDnDsyevjBatched(h, jobz, uplo, n, a, lda,
		lambda, work, work_size, info, params, batch_size));
}

inline void
syevj_buffer_size(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo,
		int n, const float* a, int lda, const float* lambda, int* work_size,
		syevj_info params)
{
	throw_if_error(cusolverDnSsyevj_bufferSize(h, jobz, uplo, n, a, lda,
		lambda, work_size, params));
}

inline void
syevj_buffer_size(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo,
		int n, const double* a, int lda, const double* lambda, int* work_size,
		syevj_info params)
{
	throw_if_error(cusolverDnDsyevj_bufferSize(h, jobz, uplo, n, a, lda,
		lambda, work_size, params));
}

inline void
syevj(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n, float* a,
		int lda, float* lambda, float* work, int work_size, int* info,
		syevj_info params)
{
	throw_if_error(cusolverDnSsyevj(h, jobz, uplo, n, a, lda,
		lambda, work, work_size, info, params));
}

inline void
syevj(dense::handle_t h, eig_mode_t jobz, fill_mode_adaptor uplo, int n,
		double* a, int lda, double* lambda, double* work, int work_size,
		int* info, syevj_info params)
{
	throw_if_error(cusolverDnDsyevj(h, jobz, uplo, n, a, lda,
		lambda, work, work_size, info, params));
}

} // namespace cusolver
