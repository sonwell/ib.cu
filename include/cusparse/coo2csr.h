#pragma once
#include "types.h"
#include "handle.h"
#include "index_base.h"

namespace cusparse {

inline void
coo2csr(handle& h, const int* rows, int nnz, int m, int* starts,
		index_base_adaptor base)
{
	throw_if_error(cusparseXcoo2csr(h, rows, nnz, m, starts, base));
}

} // namespace cusparse