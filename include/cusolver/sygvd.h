#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "fill_mode.h"

namespace cusolver {

inline void
sygvd_buffer_size(dense::handle_t h, eig_type_t itype, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, const float* a, int lda, const float* b,
		int ldb, const float* lambda, int* work_size)
{
	throw_if_error(cusolverDnSsygvd_bufferSize(h, itype, jobz, uplo, n, a,
		lda, b, ldb, lambda, work_size));
}

inline void
sygvd_buffer_size(dense::handle_t h, eig_type_t itype, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, const double* a, int lda,
		const double* b, int ldb, const double* lambda, int* work_size)
{
	throw_if_error(cusolverDnDsygvd_bufferSize(h, itype, jobz, uplo, n, a,
		lda, b, ldb, lambda, work_size));
}

inline void
sygvd(dense::handle_t h, eig_type_t itype, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, float* a, int lda, float* b, int ldb,
		float* lambda, float* work, int work_size, int* info)
{
	throw_if_error(cusolverDnSsygvd(h, itype, jobz, uplo, n, a, lda, b, ldb,
		lambda, work, work_size, info));
}

inline void
sygvd(dense::handle_t h, eig_type_t itype, eig_mode_t jobz,
		fill_mode_adaptor uplo, int n, double* a, int lda, double* b, int ldb,
		double* lambda, double* work, int work_size, int* info)
{
	throw_if_error(cusolverDnDsygvd(h, itype, jobz, uplo, n, a, lda, b, ldb,
		lambda, work, work_size, info));
}

} // namespace cusolver
