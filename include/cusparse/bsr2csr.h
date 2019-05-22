#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "direction.h"

namespace cusparse {

inline void
bsr2csr(handle& h, direction_adaptor dir_a, int mb, int nb,
		const matrix_description descr_a, const float* values_a,
		const int* starts_a, const int* indices_a, int block_dim,
		const matrix_description descr_c, float* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseSbsr2csr(h, dir_a, mb, nb, descr_a, values_a,
		starts_a, indices_a, block_dim, descr_c, values_c, starts_c,
		indices_c));
}

inline void
bsr2csr(handle& h, direction_adaptor dir_a, int mb, int nb,
		const matrix_description descr_a, const double* values_a,
		const int* starts_a, const int* indices_a, int block_dim,
		const matrix_description descr_c, double* values_c, int* starts_c,
		int* indices_c)
{
	throw_if_error(cusparseDbsr2csr(h, dir_a, mb, nb, descr_a, values_a,
		starts_a, indices_a, block_dim, descr_c, values_c, starts_c,
		indices_c));
}

} // namespace cusparse
