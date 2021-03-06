#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "csrqr_info.h"
#include "matrix_descr.h"

namespace cusolver {

inline void
csrqrsv_batched(sparse::handle_t h, int m, int n, int nnz,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, const float* b, float* x,
		int batch_size, csrqr_info info, void* buffer)
{
	throw_if_error(cusolverSpScsrqrsvBatched(h, m, n, nnz, descr_a,
		values_a, starts_a, indices_a, b, x, batch_size, info, buffer));
}

inline void
csrqrsv_batched(sparse::handle_t h, int m, int n, int nnz,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, const double* b, double* x,
		int batch_size, csrqr_info info, void* buffer)
{
	throw_if_error(cusolverSpDcsrqrsvBatched(h, m, n, nnz, descr_a,
		values_a, starts_a, indices_a, b, x, batch_size, info, buffer));
}

} // namespace cusolver
