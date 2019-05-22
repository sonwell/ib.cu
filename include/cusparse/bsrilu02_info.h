#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(bsrilu02_info_t& info)
{
	throw_if_error(cusparseCreateBsrilu02Info(&info));
}

inline void
destroy(bsrilu02_info_t& info)
{
	throw_if_error(cusparseDestroyBsrilu02Info(info));
}

using bsrilu02_info = type_wrapper<bsrilu02_info_t>;

} // namespace cusparse
