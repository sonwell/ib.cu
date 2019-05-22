#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "matrix_descr.h"

namespace cusolver {

inline void
csreigvsi(sparse::handle_t h, int m, int nnz, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		float mu0, const float* x0, int max_iter, float eps, float* mu,
		float* x)
{
	throw_if_error(cusolverSpScsreigvsi(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, mu0, x0, max_iter, eps, mu, x));
}

inline void
csreigvsi(sparse::handle_t h, int m, int nnz, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		double mu0, const double* x0, int max_iter, double eps, double* mu,
		double* x)
{
	throw_if_error(cusolverSpDcsreigvsi(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, mu0, x0, max_iter, eps, mu, x));
}

} // namespace cusolver
