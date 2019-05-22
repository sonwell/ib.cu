#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(csric02_info_t& info)
{
	throw_if_error(cusparseCreateCsric02Info(&info));
}

inline void
destroy(csric02_info_t& info)
{
	throw_if_error(cusparseDestroyCsric02Info(info));
}

using csric02_info = type_wrapper<csric02_info_t>;

} // namespace cusparse
