#pragma once
#include "util/adaptor.h"
#include "cuda/diagonal_type.h"
#include "types.h"

namespace cublas {

using cuda::diagonal_type;
using diagonal_type_adaptor = util::adaptor<
	util::enum_container<diagonal_type_t,
			CUSPARSE_DIAG_TYPE_NON_UNIT,
			CUSPARSE_DIAG_TYPE_UNIT>,
	util::enum_container<diagonal_type,
			diagonal_type::non_unit,
			diagonal_type::unit>>;

}
