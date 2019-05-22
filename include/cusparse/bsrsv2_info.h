#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(bsrsv2_info_t& info)
{
	throw_if_error(cusparseCreateBsrsv2Info(&info));
}

inline void
destroy(bsrsv2_info_t& info)
{
	throw_if_error(cusparseDestroyBsrsv2Info(info));
}

using bsrsv2_info = type_wrapper<bsrsv2_info_t>;

} // namespace cusparse
