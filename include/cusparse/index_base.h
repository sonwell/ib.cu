#pragma once
#include "util/adaptor.h"
#include "linalg/index_base.h"
#include "types.h"

namespace cusparse {

using index_base_adaptor = util::adaptor<
	util::enum_container<index_base_t,
			CUSPARSE_INDEX_BASE_ZERO,
			CUSPARSE_INDEX_BASE_ONE>,
	util::enum_container<linalg::index_base,
			linalg::index_base::zero,
			linalg::index_base::one>>;

inline index_base_adaptor
get_index_base(mat_descr_t& descr)
{
	auto res = cusparseGetMatIndexBase(descr);
	return index_base_adaptor(res);
}

inline void
set_index_base(mat_descr_t& desc, index_base_adaptor base)
{
	cusparseSetMatIndexBase(desc, base);
}

} // namespace cusparse
