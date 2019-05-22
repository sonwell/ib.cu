#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "gesvdj_info.h"

namespace cusolver {

inline void
gesvdj_batched_buffer_size(dense::handle_t h, eig_mode_t jobz, int m, int n,
		const float* a, int lda, const float* sigma, const float* u, int ldu,
		const float* v, int ldv, int* work_size, gesvdj_info params,
		int batch_size)
{
	throw_if_error(cusolverDnSgesvdjBatched_bufferSize(h, jobz, m, n, a,
		lda, sigma, u, ldu, v, ldv, work_size, params, batch_size));
}

inline void
gesvdj_batched_buffer_size(dense::handle_t h, eig_mode_t jobz, int m, int n,
		const double* a, int lda, const double* sigma, const double* u, int ldu,
		const double* v, int ldv, int* work_size, gesvdj_info params,
		int batch_size)
{
	throw_if_error(cusolverDnDgesvdjBatched_bufferSize(h, jobz, m, n, a,
		lda, sigma, u, ldu, v, ldv, work_size, params, batch_size));
}

inline void
gesvdj_batched(dense::handle_t h, eig_mode_t jobz, int m, int n, float* a,
		int lda, float* sigma, float* u, int ldu, float* v, int ldv,
		float* work, int work_size, int* info, gesvdj_info params,
		int batch_size)
{
	throw_if_error(cusolverDnSgesvdjBatched(h, jobz, m, n, a, lda, sigma, u,
		ldu, v, ldv, work, work_size, info, params, batch_size));
}

inline void
gesvdj_batched(dense::handle_t h, eig_mode_t jobz, int m, int n, double* a,
		int lda, double* sigma, double* u, int ldu, double* v, int ldv,
		double* work, int work_size, int* info, gesvdj_info params,
		int batch_size)
{
	throw_if_error(cusolverDnDgesvdjBatched(h, jobz, m, n, a, lda, sigma, u,
		ldu, v, ldv, work, work_size, info, params, batch_size));
}

inline void
gesvdj_buffer_size(dense::handle_t h, eig_mode_t jobz, int econ, int m, int n,
		const float* a, int lda, const float* sigma, const float* u, int ldu,
		const float* v, int ldv, int* work_size, gesvdj_info params)
{
	throw_if_error(cusolverDnSgesvdj_bufferSize(h, jobz, econ, m, n, a, lda,
		sigma, u, ldu, v, ldv, work_size, params));
}

inline void
gesvdj_buffer_size(dense::handle_t h, eig_mode_t jobz, int econ, int m, int n,
		const double* a, int lda, const double* sigma, const double* u, int ldu,
		const double* v, int ldv, int* work_size, gesvdj_info params)
{
	throw_if_error(cusolverDnDgesvdj_bufferSize(h, jobz, econ, m, n, a, lda,
		sigma, u, ldu, v, ldv, work_size, params));
}

inline void
gesvdj(dense::handle_t h, eig_mode_t jobz, int econ, int m, int n, float* a,
		int lda, float* sigma, float* u, int ldu, float* v, int ldv,
		float* work, int work_size, int* info, gesvdj_info params)
{
	throw_if_error(cusolverDnSgesvdj(h, jobz, econ, m, n, a, lda, sigma, u,
		ldu, v, ldv, work, work_size, info, params));
}

inline void
gesvdj(dense::handle_t h, eig_mode_t jobz, int econ, int m, int n, double* a,
		int lda, double* sigma, double* u, int ldu, double* v, int ldv,
		double* work, int work_size, int* info, gesvdj_info params)
{
	throw_if_error(cusolverDnDgesvdj(h, jobz, econ, m, n, a, lda, sigma, u,
		ldu, v, ldv, work, work_size, info, params));
}

} // namespace cusolver
