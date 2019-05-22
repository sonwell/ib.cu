#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusolver {

inline void
csrlsvqr(sparse::handle_t h, int m, int nnz, const matrix_description descr_a,
		const float* values, const int* starts, const int* indices,
		const float* b, float tol, int reorder, float* x, int* singularity)
{
	throw_if_error(cusolverSpScsrlsvqr(h, m, nnz, descr_a, values, starts,
		indices, b, tol, reorder, x, singularity));
}

inline void
csrlsvqr(sparse::handle_t h, int m, int nnz, const matrix_description descr_a,
		const double* values, const int* starts, const int* indices,
		const double* b, double tol, int reorder, double* x, int* singularity)
{
	throw_if_error(cusolverSpDcsrlsvqr(h, m, nnz, descr_a, values, starts,
		indices, b, tol, reorder, x, singularity));
}

} // namespace cusolver
