#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"

namespace cusparse {

inline void
dense2hyb(handle& h, int m, int n, const matrix_description descr_a,
		const float* a, int lda, const int* nnz_per_row, hyb_matrix hyb_a,
		int ell_width, hyp_partition_t partition)
{
	throw_if_error(cusparseSdense2hyb(h, m, n, descr_a, a, lda, nnz_per_row,
		hyb_a, ell_width, partition));
}

inline void
dense2hyb(handle& h, int m, int n, const matrix_description descr_a,
		const double* a, int lda, const int* nnz_per_row, hyb_matrix hyb_a,
		int ell_width, hyp_partition_t partition)
{
	throw_if_error(cusparseDdense2hyb(h, m, n, descr_a, a, lda, nnz_per_row,
		hyb_a, ell_width, partition));
}

} // namespace cusparse
