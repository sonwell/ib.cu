#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "color_info.h"

namespace cusparse {

inline void
csrcolor(handle& h, int m, int nnz, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		const float* fraction, int* ncolors, int* coloring, int* reordering,
		const color_info_t info)
{
	throw_if_error(cusparseScsrcolor(h, m, nnz, descr_a, values_a, starts_a,
		indices_a, fraction, ncolors, coloring, reordering, info));
}

inline void
csrcolor(handle& h, int m, int nnz, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		const double* fraction, int* ncolors, int* coloring, int* reordering,
		const color_info_t info)
{
	throw_if_error(cusparseDcsrcolor(h, m, nnz, descr_a, values_a, starts_a,
		indices_a, fraction, ncolors, coloring, reordering, info));
}

} // namespace cusparse
