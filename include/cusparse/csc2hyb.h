#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"

namespace cusparse {

inline void
csc2hyb(handle& h, int m, int n, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		hyb_matrix hyb_a, int ell_width, hyp_partition_t partition)
{
	throw_if_error(cusparseScsc2hyb(h, m, n, descr_a, values_a, starts_a,
		indices_a, hyb_a, ell_width, partition));
}

inline void
csc2hyb(handle& h, int m, int n, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		hyb_matrix hyb_a, int ell_width, hyp_partition_t partition)
{
	throw_if_error(cusparseDcsc2hyb(h, m, n, descr_a, values_a, starts_a,
		indices_a, hyb_a, ell_width, partition));
}

} // namespace cusparse
