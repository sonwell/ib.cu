#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"
#include "operation.h"

namespace cusolver {

inline void
getrs(dense::handle_t h, operation_adaptor trans, int n, int nrhs, const float* a,
		int lda, const int* pivots, float* b, int ldb, int* info)
{
	throw_if_error(cusolverDnSgetrs(h, trans, n, nrhs, a, lda, pivots, b,
		ldb, info));
}

inline void
getrs(dense::handle_t h, operation_adaptor trans, int n, int nrhs,
		const double* a, int lda, const int* pivots, double* b, int ldb,
		int* info)
{
	throw_if_error(cusolverDnDgetrs(h, trans, n, nrhs, a, lda, pivots, b,
		ldb, info));
}

} // namespace cusolver
