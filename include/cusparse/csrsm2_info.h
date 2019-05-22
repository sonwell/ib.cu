#pragma once
#include "types.h"
#include "excpeptions.h"

namespace cusparse {

inline void
create(csrsm2_info_t& info)
{
	throw_if_error(cusparseCreateCsrsm2Info(&info));
}

inline void
destroy(csrsm2_info_t& info)
{
	throw_if_error(cusparseDestroyCsrsm2Info(info));
}

using csrsm2_info = type_wrapper<csrsm2_info_t>;

} // namespace cusparse
