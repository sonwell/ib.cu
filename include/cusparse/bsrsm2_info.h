#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(bsrsm2_info_t& info)
{
	throw_if_error(cusparseCreateBsrsm2Info(&info));
}

inline void
destroy(bsrsm2_info_t& info)
{
	throw_if_error(cusparseDestroyBsrsm2Info(info));
}

using bsrsm2_info = type_wrapper<bsrsm2_info_t>;

} // namespace cusparse
