#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusparse {

inline void
csrgeam_nnz(handle& h, int m, int n, const matrix_description descr_a,
		int nnz_a, const int* starts_a, const int* indices_a,
		const matrix_description descr_b, int nnz_b, const int* starts_b,
		const int* indices_b, const matrix_description descr_c, int* starts_c,
		int* nnz_total)
{
	throw_if_error(cusparseXcsrgeamNnz(h, m, n, descr_a, nnz_a, starts_a,
		indices_a, descr_b, nnz_b, starts_b, indices_b, descr_c, starts_c,
		nnz_total), "csrgeam_nnz failed");
}

inline void
csrgeam(handle& h, int m, int n, const float* alpha,
		const matrix_description descr_a, int nnz_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* beta,
		const matrix_description descr_b, int nnz_b, const float* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, float* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseScsrgeam(h, m, n, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, beta, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c));
}

inline void
csrgeam(handle& h, int m, int n, const double* alpha,
		const matrix_description descr_a, int nnz_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* beta,
		const matrix_description descr_b, int nnz_b, const double* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, double* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseDcsrgeam(h, m, n, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, beta, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c),
			"csrgeam failed");
}

#if CUDART_VERSION > 9000

inline void
csrgeam2_buffer_size_ext(handle& h, int m, int n, const float* alpha,
		const matrix_description descr_a, int nnz_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* beta,
		const matrix_description descr_b, int nnz_b, const float* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, const float* values_c,
		const int* starts_c, const int* indices_c, size_t* buffer_size)
{
	throw_if_error(cusparseScsrgeam2_bufferSizeExt(h, m, n, alpha, descr_a,
		nnz_a, values_a, starts_a, indices_a, beta, descr_b, nnz_b,
		values_b, starts_b, indices_b, descr_c, values_c, starts_c,
		indices_c, buffer_size));
}

inline void
csrgeam2_buffer_size_ext(handle& h, int m, int n, const double* alpha,
		const matrix_description descr_a, int nnz_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* beta,
		const matrix_description descr_b, int nnz_b, const double* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, const double* values_c,
		const int* starts_c, const int* indices_c, size_t* buffer_size)
{
	throw_if_error(cusparseDcsrgeam2_bufferSizeExt(h, m, n, alpha, descr_a,
		nnz_a, values_a, starts_a, indices_a, beta, descr_b, nnz_b,
		values_b, starts_b, indices_b, descr_c, values_c, starts_c,
		indices_c, buffer_size));
}

inline void
csrgeam2_nnz(handle& h, int m, int n, const matrix_description descr_a,
		int nnz_a, const int* starts_a, const int* indices_a,
		const matrix_description descr_b, int nnz_b, const int* starts_b,
		const int* indices_b, const matrix_description descr_c, int* starts_c,
		int* nnz_total, void* workspace)
{
	throw_if_error(cusparseXcsrgeam2Nnz(h, m, n, descr_a, nnz_a, starts_a,
		indices_a, descr_b, nnz_b, starts_b, indices_b, descr_c, starts_c,
		nnz_total, workspace));
}

inline void
csrgeam2(handle& h, int m, int n, const float* alpha,
		const matrix_description descr_a, int nnz_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* beta,
		const matrix_description descr_b, int nnz_b, const float* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, float* values_c, int* starts_c,
		int* indices_c, void* buffer)
{
	throw_if_error(cusparseScsrgeam2(h, m, n, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, beta, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c, buffer));
}

inline void
csrgeam2(handle& h, int m, int n, const double* alpha,
		const matrix_description descr_a, int nnz_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* beta,
		const matrix_description descr_b, int nnz_b, const double* values_b,
		const int* starts_b, const int* indices_b,
		const matrix_description descr_c, double* values_c, int* starts_c,
		int* indices_c, void* buffer)
{
	throw_if_error(cusparseDcsrgeam2(h, m, n, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, beta, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c, buffer));
}

#endif

} // namespace cusparse
