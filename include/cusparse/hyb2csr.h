#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"

namespace cusparse {

inline void
hyb2csr(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		float* values_a, int* starts_a, int* indices_a)
{
	throw_if_error(cusparseShyb2csr(h, descr_a, hyb_a, values_a, starts_a,
		indices_a));
}

inline void
hyb2csr(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		double* values_a, int* starts_a, int* indices_a)
{
	throw_if_error(cusparseDhyb2csr(h, descr_a, hyb_a, values_a, starts_a,
		indices_a));
}

} // namespace cusparse
