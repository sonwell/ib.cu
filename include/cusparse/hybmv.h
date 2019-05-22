#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"
#include "operation.h"

namespace cusparse {

inline void
hybmv(handle& h, operation_adaptor trans_a, const float* alpha,
		const matrix_description descr_a, const hyb_matrix hyb_a,
		const float* x, const float* beta, float* y)
{
	throw_if_error(cusparseShybmv(h, trans_a, alpha, descr_a, hyb_a, x,
		beta, y));
}

inline void
hybmv(handle& h, operation_adaptor trans_a, const double* alpha,
		const matrix_description descr_a, const hyb_matrix hyb_a,
		const double* x, const double* beta, double* y)
{
	throw_if_error(cusparseDhybmv(h, trans_a, alpha, descr_a, hyb_a, x,
		beta, y));
}

} // namespace cusparse
