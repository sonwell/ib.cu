#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "prune_info.h"

namespace cusparse {

inline void
prune_dense2csr_buffer_size_ext(handle& h, int m, int n, const float* a,
		int lda, const float* threshold, const matrix_description descr_c,
		const float* values_c, const int* starts_c, const int* indices_c,
		size_t* buffer_size)
{
	throw_if_error(cusparseSpruneDense2csr_bufferSizeExt(h, m, n, a, lda,
		threshold, descr_c, values_c, starts_c, indices_c, buffer_size));
}

inline void
prune_dense2csr_buffer_size_ext(handle& h, int m, int n, const double* a,
		int lda, const double* threshold, const matrix_description descr_c,
		const double* values_c, const int* starts_c, const int* indices_c,
		size_t* buffer_size)
{
	throw_if_error(cusparseDpruneDense2csr_bufferSizeExt(h, m, n, a, lda,
		threshold, descr_c, values_c, starts_c, indices_c, buffer_size));
}

inline void
prune_dense2csr_nnz(handle& h, int m, int n, const float* a, int lda,
		const float* threshold, const matrix_description descr_c, int* starts_c,
		int* nnz_total, void* buffer)
{
	throw_if_error(cusparseSpruneDense2csrNnz(h, m, n, a, lda, threshold,
		descr_c, starts_c, nnz_total, buffer));
}

inline void
prune_dense2csr_nnz(handle& h, int m, int n, const double* a, int lda,
		const double* threshold, const matrix_description descr_c,
		int* starts_c, int* nnz_total, void* buffer)
{
	throw_if_error(cusparseDpruneDense2csrNnz(h, m, n, a, lda, threshold,
		descr_c, starts_c, nnz_total, buffer));
}

inline void
prune_dense2csr(handle& h, int m, int n, const float* a, int lda,
		const float* threshold, const matrix_description descr_c,
		float* values_c, const int* starts_c, int* indices_c, void* buffer)
{
	throw_if_error(cusparseSpruneDense2csr(h, m, n, a, lda, threshold,
		descr_c, values_c, starts_c, indices_c, buffer));
}

inline void
prune_dense2csr(handle& h, int m, int n, const double* a, int lda,
		const double* threshold, const matrix_description descr_c,
		double* values_c, const int* starts_c, int* indices_c, void* buffer)
{
	throw_if_error(cusparseDpruneDense2csr(h, m, n, a, lda, threshold,
		descr_c, values_c, starts_c, indices_c, buffer));
}

inline void
prune_csr2csr_buffer_size_ext(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* threshold,
		const matrix_description descr_c, const float* values_c,
		const int* starts_c, const int* indices_c, size_t* buffer_size)
{
	throw_if_error(cusparseSpruneCsr2csr_bufferSizeExt(h, m, n, nnz_a,
		descr_a, values_a, starts_a, indices_a, threshold, descr_c,
		values_c, starts_c, indices_c, buffer_size));
}

inline void
prune_csr2csr_buffer_size_ext(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* threshold,
		const matrix_description descr_c, const double* values_c,
		const int* starts_c, const int* indices_c, size_t* buffer_size)
{
	throw_if_error(cusparseDpruneCsr2csr_bufferSizeExt(h, m, n, nnz_a,
		descr_a, values_a, starts_a, indices_a, threshold, descr_c,
		values_c, starts_c, indices_c, buffer_size));
}

inline void
prune_csr2csr_nnz(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* threshold,
		const matrix_description descr_c, int* starts_c, int* nnz_total,
		void* buffer)
{
	throw_if_error(cusparseSpruneCsr2csrNnz(h, m, n, nnz_a, descr_a,
		values_a, starts_a, indices_a, threshold, descr_c, starts_c,
		nnz_total, buffer));
}

inline void
prune_csr2csr_nnz(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* threshold,
		const matrix_description descr_c, int* starts_c, int* nnz_total,
		void* buffer)
{
	throw_if_error(cusparseDpruneCsr2csrNnz(h, m, n, nnz_a, descr_a,
		values_a, starts_a, indices_a, threshold, descr_c, starts_c,
		nnz_total, buffer));
}

inline void
prune_csr2csr(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* threshold,
		const matrix_description descr_c, float* values_c, const int* starts_c,
		int* indices_c, void* buffer)
{
	throw_if_error(cusparseSpruneCsr2csr(h, m, n, nnz_a, descr_a, values_a,
		starts_a, indices_a, threshold, descr_c, values_c, starts_c,
		indices_c, buffer));
}

inline void
prune_csr2csr(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* threshold,
		const matrix_description descr_c, double* values_c, const int* starts_c,
		int* indices_c, void* buffer)
{
	throw_if_error(cusparseDpruneCsr2csr(h, m, n, nnz_a, descr_a, values_a,
		starts_a, indices_a, threshold, descr_c, values_c, starts_c,
		indices_c, buffer));
}

inline void
prune_dense2csr_by_percentage_buffer_size_ext(handle& h, int m, int n,
		const float* a, int lda, float percentage,
		const matrix_description descr_c, const float* values_c,
		const int* starts_c, const int* indices_c, prune_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSpruneDense2csrByPercentage_bufferSizeExt(h, m,
		n, a, lda, percentage, descr_c, values_c, starts_c, indices_c, info,
		buffer_size));
}

inline void
prune_dense2csr_by_percentage_buffer_size_ext(handle& h, int m, int n,
		const double* a, int lda, float percentage,
		const matrix_description descr_c, const double* values_c,
		const int* starts_c, const int* indices_c, prune_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDpruneDense2csrByPercentage_bufferSizeExt(h, m,
		n, a, lda, percentage, descr_c, values_c, starts_c, indices_c, info,
		buffer_size));
}

inline void
prune_dense2csr_nnz_by_percentage(handle& h, int m, int n, const float* a,
		int lda, float percentage, const matrix_description descr_c,
		int* starts_c, int* nnz_total, prune_info info, void* buffer)
{
	throw_if_error(cusparseSpruneDense2csrNnzByPercentage(h, m, n, a, lda,
		percentage, descr_c, starts_c, nnz_total, info, buffer));
}

inline void
prune_dense2csr_nnz_by_percentage(handle& h, int m, int n, const double* a,
		int lda, float percentage, const matrix_description descr_c,
		int* starts_c, int* nnz_total, prune_info info, void* buffer)
{
	throw_if_error(cusparseDpruneDense2csrNnzByPercentage(h, m, n, a, lda,
		percentage, descr_c, starts_c, nnz_total, info, buffer));
}

inline void
prune_dense2csr_by_percentage(handle& h, int m, int n, const float* a, int lda,
		float percentage, const matrix_description descr_c, float* values_c,
		const int* starts_c, int* indices_c, prune_info info, void* buffer)
{
	throw_if_error(cusparseSpruneDense2csrByPercentage(h, m, n, a, lda,
		percentage, descr_c, values_c, starts_c, indices_c, info, buffer));
}

inline void
prune_dense2csr_by_percentage(handle& h, int m, int n, const double* a, int lda,
		float percentage, const matrix_description descr_c, double* values_c,
		const int* starts_c, int* indices_c, prune_info info, void* buffer)
{
	throw_if_error(cusparseDpruneDense2csrByPercentage(h, m, n, a, lda,
		percentage, descr_c, values_c, starts_c, indices_c, info, buffer));
}

inline void
prune_csr2csr_by_percentage_buffer_size_ext(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, float percentage,
		const matrix_description descr_c, const float* values_c,
		const int* starts_c, const int* indices_c, prune_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseSpruneCsr2csrByPercentage_bufferSizeExt(h, m, n,
		nnz_a, descr_a, values_a, starts_a, indices_a, percentage, descr_c,
		values_c, starts_c, indices_c, info, buffer_size));
}

inline void
prune_csr2csr_by_percentage_buffer_size_ext(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, float percentage,
		const matrix_description descr_c, const double* values_c,
		const int* starts_c, const int* indices_c, prune_info info,
		size_t* buffer_size)
{
	throw_if_error(cusparseDpruneCsr2csrByPercentage_bufferSizeExt(h, m, n,
		nnz_a, descr_a, values_a, starts_a, indices_a, percentage, descr_c,
		values_c, starts_c, indices_c, info, buffer_size));
}

inline void
prune_csr2csr_nnz_by_percentage(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, float percentage,
		const matrix_description descr_c, int* starts_c, int* nnz_total,
		prune_info info, void* buffer)
{
	throw_if_error(cusparseSpruneCsr2csrNnzByPercentage(h, m, n, nnz_a,
		descr_a, values_a, starts_a, indices_a, percentage, descr_c,
		starts_c, nnz_total, info, buffer));
}

inline void
prune_csr2csr_nnz_by_percentage(handle& h, int m, int n, int nnz_a,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, float percentage,
		const matrix_description descr_c, int* starts_c, int* nnz_total,
		prune_info info, void* buffer)
{
	throw_if_error(cusparseDpruneCsr2csrNnzByPercentage(h, m, n, nnz_a,
		descr_a, values_a, starts_a, indices_a, percentage, descr_c,
		starts_c, nnz_total, info, buffer));
}

} // namespace cusparse
