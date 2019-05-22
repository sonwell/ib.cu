#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"

namespace cusparse {

inline void
hyb2csc(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		float* values, int* starts, int* indices)
{
	throw_if_error(cusparseShyb2csc(h, descr_a, hyb_a, values, starts,
		indices));
}

inline void
hyb2csc(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		double* values, int* starts, int* indices)
{
	throw_if_error(cusparseDhyb2csc(h, descr_a, hyb_a, values, starts,
		indices));
}

} // namespace cusparse
