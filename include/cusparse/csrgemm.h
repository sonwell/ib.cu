#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "operation.h"
#include "csrgemm2_info.h"

namespace cusparse {

inline void
csrgemm_nnz(handle& h, operation_adaptor trans_a, operation_adaptor trans_b,
		int m, int n, int k, const matrix_description descr_a, const int nnz_a,
		const int* starts_a, const int* indices_a,
		const matrix_description descr_b, const int nnz_b, const int* starts_b,
		const int* indices_b, const matrix_description descr_c, int* starts_c,
		int* nnz_total)
{
	throw_if_error(cusparseXcsrgemmNnz(h, trans_a, trans_b, m, n, k,
		descr_a, nnz_a, starts_a, indices_a, descr_b, nnz_b, starts_b,
		indices_b, descr_c, starts_c, nnz_total));
}

inline void
csrgemm(handle& h, operation_adaptor trans_a, operation_adaptor trans_b, int m,
		int n, int k, const matrix_description descr_a, const int nnz_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const matrix_description descr_b, const int nnz_b,
		const float* values_b, const int* starts_b, const int* indices_b,
		const matrix_description descr_c, float* values_c, const int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseScsrgemm(h, trans_a, trans_b, m, n, k, descr_a,
		nnz_a, values_a, starts_a, indices_a, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c));
}

inline void
csrgemm(handle& h, operation_adaptor trans_a, operation_adaptor trans_b, int m,
		int n, int k, const matrix_description descr_a, const int nnz_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const matrix_description descr_b, const int nnz_b,
		const double* values_b, const int* starts_b, const int* indices_b,
		const matrix_description descr_c, double* values_c, const int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseDcsrgemm(h, trans_a, trans_b, m, n, k, descr_a,
		nnz_a, values_a, starts_a, indices_a, descr_b, nnz_b, values_b,
		starts_b, indices_b, descr_c, values_c, starts_c, indices_c));
}

inline void
csrgemm2_buffer_size_ext(handle& h, int m, int n, int k, const float* alpha,
		const matrix_description descr_a, int nnz_a, const int* starts_a,
		const int* indices_a, const matrix_description descr_b, int nnz_b,
		const int* starts_b, const int* indices_b, const float* beta,
		const matrix_description descr_d, int nnz_d, const int* starts_d,
		const int* indices_d, csrgemm2_info info, size_t* buffer_size)
{
	throw_if_error(cusparseScsrgemm2_bufferSizeExt(h, m, n, k, alpha,
		descr_a, nnz_a, starts_a, indices_a, descr_b, nnz_b, starts_b,
		indices_b, beta, descr_d, nnz_d, starts_d, indices_d, info,
		buffer_size));
}

inline void
csrgemm2_buffer_size_ext(handle& h, int m, int n, int k, const double* alpha,
		const matrix_description descr_a, int nnz_a, const int* starts_a,
		const int* indices_a, const matrix_description descr_b, int nnz_b,
		const int* starts_b, const int* indices_b, const double* beta,
		const matrix_description descr_d, int nnz_d, const int* starts_d,
		const int* indices_d, csrgemm2_info info, size_t* buffer_size)
{
	throw_if_error(cusparseDcsrgemm2_bufferSizeExt(h, m, n, k, alpha,
		descr_a, nnz_a, starts_a, indices_a, descr_b, nnz_b, starts_b,
		indices_b, beta, descr_d, nnz_d, starts_d, indices_d, info,
		buffer_size));
}

inline void
csrgemm2_nnz(handle& h, int m, int n, int k, const matrix_description descr_a,
		int nnz_a, const int* starts_a, const int* indices_a,
		const matrix_description descr_b, int nnz_b, const int* starts_b,
		const int* indices_b, const matrix_description descr_d, int nnz_d,
		const int* starts_d, const int* indices_d,
		const matrix_description descr_c, int* starts_c, int* nnz_total,
		const csrgemm2_info info, void* buffer)
{
	throw_if_error(cusparseXcsrgemm2Nnz(h, m, n, k, descr_a, nnz_a,
		starts_a, indices_a, descr_b, nnz_b, starts_b, indices_b, descr_d,
		nnz_d, starts_d, indices_d, descr_c, starts_c, nnz_total, info,
		buffer));
}

inline void
csrgemm2(handle& h, int m, int n, int k, const float* alpha,
		const matrix_description descr_a, int nnz_a, const float* values_a,
		const int* starts_a, const int* indices_a,
		const matrix_description descr_b, int nnz_b, const float* values_b,
		const int* starts_b, const int* indices_b, const float* beta,
		const matrix_description descr_d, int nnz_d, const float* values_d,
		const int* starts_d, const int* indices_d,
		const matrix_description descr_c, float* values_c, const int* starts_c,
		int* indices_c, const csrgemm2_info info, void* buffer)
{
	throw_if_error(cusparseScsrgemm2(h, m, n, k, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, descr_b, nnz_b, values_b, starts_b,
		indices_b, beta, descr_d, nnz_d, values_d, starts_d, indices_d,
		descr_c, values_c, starts_c, indices_c, info, buffer));
}

inline void
csrgemm2(handle& h, int m, int n, int k, const double* alpha,
		const matrix_description descr_a, int nnz_a, const double* values_a,
		const int* starts_a, const int* indices_a,
		const matrix_description descr_b, int nnz_b, const double* values_b,
		const int* starts_b, const int* indices_b, const double* beta,
		const matrix_description descr_d, int nnz_d, const double* values_d,
		const int* starts_d, const int* indices_d,
		const matrix_description descr_c, double* values_c, const int* starts_c,
		int* indices_c, const csrgemm2_info info, void* buffer)
{
	throw_if_error(cusparseDcsrgemm2(h, m, n, k, alpha, descr_a, nnz_a,
		values_a, starts_a, indices_a, descr_b, nnz_b, values_b, starts_b,
		indices_b, beta, descr_d, nnz_d, values_d, starts_d, indices_d,
		descr_c, values_c, starts_c, indices_c, info, buffer));
}

} // namespace cusparse
