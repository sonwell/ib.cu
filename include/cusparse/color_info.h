#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(color_info_t& info)
{
	throw_if_error(cusparseCreateColorInfo(&info));
}

inline void
destroy(color_info_t& info)
{
	throw_if_error(cusparseDestroyColorInfo(info));
}

using color_info = type_wrapper<color_info_t>;

} // namespace cusparse
