#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(csrgemm2_info_t& info)
{
	throw_if_error(cusparseCreateCsrgemm2Info(&info));
}

inline void
destroy(csrgemm2_info_t& info)
{
	throw_if_error(cusparseDestroyCsrgemm2Info(info));
}

using csrgemm2_info = type_wrapper<csrgemm2_info_t>;

} // namespace cusparse
