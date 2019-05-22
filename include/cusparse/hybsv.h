#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "hyb_matrix_descr.h"
#include "solve_analysis_info.h"
#include "operation.h"

namespace cusparse {

inline void
hybsv_analysis(handle& h, operation_adaptor trans_a,
		const matrix_description descr_a, hyb_matrix hyb_a,
		solve_analysis_info info)
{
	throw_if_error(cusparseShybsv_analysis(h, trans_a, descr_a, hyb_a, info));
}

inline void
hybsv_analysis(handle& h, operation_adaptor trans_a,
		const matrix_description descr_a, hyb_matrix hyb_a,
		solve_analysis_info info)
{
	throw_if_error(cusparseDhybsv_analysis(h, trans_a, descr_a, hyb_a, info));
}

inline void
hybsv_solve(handle& h, operation_adaptor trans, const float* alpha,
		const matrix_description descr_a, const hyb_matrix hyb_a,
		solve_analysis_info info, const float* f, float* x)
{
	throw_if_error(cusparseShybsv_solve(h, trans, alpha, descr_a, hyb_a,
		info, f, x));
}

inline void
hybsv_solve(handle& h, operation_adaptor trans, const double* alpha,
		const matrix_description descr_a, const hyb_matrix hyb_a,
		solve_analysis_info info, const double* f, double* x)
{
	throw_if_error(cusparseDhybsv_solve(h, trans, alpha, descr_a, hyb_a,
		info, f, x));
}

} // namespace cusparse
