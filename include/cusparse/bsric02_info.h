#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(bsric02_info_t& info)
{
	throw_if_error(cusparseCreateCsrsv2Info(&info));
}

inline void
destroy(bsric02_info_t& info)
{
	throw_if_error(cusparseDestroyCsrsv2Info(info));

}

using bsric02_info = type_wrapper<bsric02_info_t>;

} // namespace cusparse
