#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(csrsv2_info_t& info)
{
	throw_if_error(cusparseCreateCsrsv2Info(&info));
}

inline void
destroy(csrsv2_info_t& info)
{
	throw_if_error(cusparseDestroyCsrsv2Info(info));
}

using csrsv2_info = type_wrapper<csrsv2_info_t>;

} // namespace cusparse
