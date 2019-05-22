#pragma once
#include <ostream>
#include "util/adaptor.h"
#include "cuda/fill_mode.h"
#include "types.h"

namespace cusparse {

using cuda::fill_mode;
using fill_mode_adaptor = util::adaptor<
	util::enum_container<fill_mode_t,
			CUSPARSE_FILL_MODE_LOWER,
			CUSPARSE_FILL_MODE_UPPER>,
	util::enum_container<fill_mode,
			fill_mode::lower,
			fill_mode::upper>>;

inline fill_mode
get_fill_mode(mat_descr_t& descr)
{
	auto res = cusparseGetMatFillMode(descr);
	return fill_mode_adaptor(res);
}

inline void
set_fill_mode(mat_descr_t& desc, fill_mode_adaptor diag)
{
	cusparseSetMatFillMode(desc, diag);
}

} // namespace cusparse
