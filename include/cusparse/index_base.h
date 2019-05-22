#pragma once
#include <ostream>
#include "util/adaptor.h"
#include "types.h"

namespace cusparse {

enum class index_base { zero, one };
using index_base_adaptor = util::adaptor<
	util::enum_container<index_base_t,
			CUSPARSE_INDEX_BASE_ZERO,
			CUSPARSE_INDEX_BASE_ONE>,
	util::enum_container<index_base,
			index_base::zero,
			index_base::one>>;

inline std::ostream&
operator<<(std::ostream& out, index_base v)
{
	switch (v) {
		case index_base::zero:
			return out << 0;
		case index_base::one:
			return out << 1;
	}
	throw util::bad_enum_value();
}

inline index_base
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
