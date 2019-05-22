#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"

namespace cusparse {

inline void
hyb2dense(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		float* a, int lda)
{
	throw_if_error(cusparseShyb2dense(h, descr_a, hyb_a, a, lda));
}

inline void
hyb2dense(handle& h, const matrix_description descr_a, const hyb_matrix hyb_a,
		double* a, int lda)
{
	throw_if_error(cusparseDhyb2dense(h, descr_a, hyb_a, a, lda));
}

} // namespace cusparse
