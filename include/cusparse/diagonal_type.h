#pragma once
#include <ostream>
#include "util/adaptor.h"
#include "cuda/diagonal_type.h"
#include "types.h"

namespace cusparse {

using cuda::diagonal_type;
using diagonal_adaptor = util::adaptor<
	util::enum_container<diagonal_type_t,
			CUSPARSE_DIAG_TYPE_NON_UNIT,
			CUSPARSE_DIAG_TYPE_UNIT>,
	util::enum_container<diagonal_type,
			diagonal_type::non_unit,
			diagonal_type::unit>>;

inline diagonal_type
get_diag_type(mat_descr_t& descr)
{
	auto res = cusparseGetMatDiagType(descr);
	return diagonal_adaptor(res);
}

inline void
set_diag_type(mat_descr_t& desc, diagonal_adaptor diag)
{
	cusparseSetMatDiagType(desc, diag);
}

} // namespace cusparse
