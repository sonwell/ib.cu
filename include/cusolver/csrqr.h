#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "csrqr_info.h"
#include "matrix_descr.h"

namespace cusolver {

inline void
csrqr_analysis_batched(sparse::handle_t h, int m, int n, int nnz_a,
		const matrix_description descr_a, const int* starts_a,
		const int* indices_a, csrqr_info info)
{
	throw_if_error(cusolverSpXcsrqrAnalysisBatched(h, m, n, nnz_a, descr_a,
		starts_a, indices_a, info));
}

inline void
csrqr_buffer_info_batched(sparse::handle_t h, int m, int n, int nnz,
		const matrix_description descr_a, const float* values,
		const int* starts, const int* indices, int batch_size, csrqr_info info,
		size_t* internal_data_bytes, size_t* workspace_bytes)
{
	throw_if_error(cusolverSpScsrqrBufferInfoBatched(h, m, n, nnz, descr_a,
		values, starts, indices, batch_size, info, internal_data_bytes,
		workspace_bytes));
}

inline void
csrqr_buffer_info_batched(sparse::handle_t h, int m, int n, int nnz,
		const matrix_description descr_a, const double* values,
		const int* starts, const int* indices, int batch_size, csrqr_info info,
		size_t* internal_data_bytes, size_t* workspace_bytes)
{
	throw_if_error(cusolverSpDcsrqrBufferInfoBatched(h, m, n, nnz, descr_a,
		values, starts, indices, batch_size, info, internal_data_bytes,
		workspace_bytes));
}

} // namespace cusolver
