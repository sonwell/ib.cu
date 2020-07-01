#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusolver {

enum class job : char {
	all = 'A',
	small = 'S',
	overwrite = 'O',
	none = 'N',
};

inline void
gesvd_buffer_size(dense::handle_t h, int m, int n, float*, int, int* work_size)
{
	throw_if_error(cusolverDnSgesvd_bufferSize(h, m, n, work_size));
}

inline void
gesvd_buffer_size(dense::handle_t h, int m, int n, double*, int, int* work_size)
{
	throw_if_error(cusolverDnDgesvd_bufferSize(h, m, n, work_size));
}

inline void
gesvd(dense::handle_t h, job jobu, job jobvt, int m, int n,
		float* a, int lda, float* sigma, float* u, int ldu, float* v, int ldvt,
		float* work, int work_size, float* rwork, int* info)
{
	throw_if_error(cusolverDnSgesvd(h, static_cast<char>(jobu),
		static_cast<char>(jobvt), m, n, a, lda, sigma, u, ldu, v,
		ldvt, work, work_size, rwork, info));
}

inline void
gesvd(dense::handle_t h, job jobu, job jobvt, int m, int n,
		double* a, int lda, double* sigma, double* u, int ldu, double* v,
		int ldvt, double* work, int work_size, double* rwork, int* info)
{
	throw_if_error(cusolverDnDgesvd(h, static_cast<char>(jobu),
		static_cast<char>(jobvt), m, n, a, lda, sigma, u, ldu,
		v, ldvt, work, work_size, rwork, info));
}

} // namespace cusolver
