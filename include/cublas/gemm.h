#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"

#if CUDART_VERSION > 9000
#define CUBLAS_GEMM_POINTER_CONSTNESS const
#else
#define CUBLAS_GEMM_POINTER_CONSTNESS
#endif

namespace cublas {

inline void
gemm(handle_t h, operation_adaptor transa, operation_adaptor transb, int m, int n,
		int k, const float* alpha, const float* a, int lda, const float* b,
		int ldb, const float* beta, float* c, int ldc)
{
	throw_if_error(cublasSgemm(h, transa, transb, m, n, k, alpha, a, lda, b,
		ldb, beta, c, ldc));
}

inline void
gemm(handle_t h, operation_adaptor transa, operation_adaptor transb, int m, int n,
		int k, const double* alpha, const double* a, int lda, const double* b,
		int ldb, const double* beta, double* c, int ldc)
{
	throw_if_error(cublasDgemm(h, transa, transb, m, n, k, alpha, a, lda, b,
		ldb, beta, c, ldc));
}

inline void
gemm_batched(handle_t h, operation_adaptor transa, operation_adaptor transb,
		int m, int n, int k, const float* alpha,
		const float* CUBLAS_GEMM_POINTER_CONSTNESS a[], int lda,
		const float* CUBLAS_GEMM_POINTER_CONSTNESS b[], int ldb,
		const float* beta, float* CUBLAS_GEMM_POINTER_CONSTNESS c[], int ldc,
		int batch_count)
{
	throw_if_error(cublasSgemmBatched(h, transa, transb, m, n, k, alpha,
		a, lda, b, ldb, beta, c, ldc, batch_count));
}

inline void
gemm_batched(handle_t h, operation_adaptor transa, operation_adaptor transb,
		int m, int n, int k, const double* alpha,
		const double* CUBLAS_GEMM_POINTER_CONSTNESS a[], int lda,
		const double* CUBLAS_GEMM_POINTER_CONSTNESS b[], int ldb,
		const double* beta, double* CUBLAS_GEMM_POINTER_CONSTNESS c[], int ldc,
		int batch_count)
{
	throw_if_error(cublasDgemmBatched(h, transa, transb, m, n, k, alpha,
		a, lda, b, ldb, beta, c, ldc, batch_count));
}

inline void
gemm_strided_batched(handle_t h, operation_adaptor transa,
		operation_adaptor transb, int m, int n, int k, const float* alpha,
		const float* a, int lda, long long int stride_a, const float* b,
		int ldb, long long int stride_b, const float* beta, float* C, int ldc,
		long long int stride_c, int batch_count)
{
	throw_if_error(cublasSgemmStridedBatched(h, transa, transb, m, n, k,
		alpha, a, lda, stride_a, b, ldb, stride_b, beta,
		C, ldc, stride_c, batch_count));
}

inline void
gemm_strided_batched(handle_t h, operation_adaptor transa,
		operation_adaptor transb, int m, int n, int k, const double* alpha,
		const double* a, int lda, long long int stride_a, const double* b,
		int ldb, long long int stride_b, const double* beta, double* C, int ldc,
		long long int stride_c, int batch_count)
{
	throw_if_error(cublasDgemmStridedBatched(h, transa, transb, m, n, k,
		alpha, a, lda, stride_a, b, ldb, stride_b, beta,
		C, ldc, stride_c, batch_count));
}

} // namespace cublas

#undef CUBLAS_GEMM_POINTER_CONSTNESS
