#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusolver {

inline void
create(csrqr_info_t& info)
{
	throw_if_error(cusolverSpCreateCsrqrInfo(&info));
}

inline void
destroy(csrqr_info_t& info)
{
	throw_if_error(cusolverSpDestroyCsrqrInfo(info));
}

using csrqr_info = type_wrapper<csrqr_info_t>;

}
