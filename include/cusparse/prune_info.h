#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(prune_info_t& info)
{
	throw_if_error(cusparseCreatePruneInfo(&info));
}

inline void
destroy(prune_info_t& info)
{
	throw_if_error(cusparseDestroyPruneInfo(info));
}

using prune_info = type_wrapper<prune_info_t>;

} // namespace cusparse
