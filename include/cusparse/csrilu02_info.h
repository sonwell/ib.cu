#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(csrilu02_info_t& info)
{
	throw_if_error(cusparseCreateCsrilu02Info(&info));
}

inline void
destroy(csrilu02_info_t& info)
{
	throw_if_error(cusparseDestroyCsrilu02Info(info));
}

using csrilu02_info = type_wrapper<csrilu02_info_t>;

} // namespace cusparse
