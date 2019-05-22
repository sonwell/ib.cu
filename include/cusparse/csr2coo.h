#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
csr2coo(handle& h, const int* starts, int nnz, int m, int* rows,
		index_base_adaptor base)
{
	throw_if_error(cusparseXcsr2coo(h, starts, nnz, m, rows, base));
}

} // namespace cusparse