#pragma once
#include "util/adaptor.h"
#include "cuda/diagonal_type.h"
#include "types.h"

namespace cublas {

using cuda::diagonal_type;
using diagonal_type_adaptor = util::adaptor<
	util::enum_container<diag_type_t,
			CUBLAS_DIAG_NON_UNIT,
			CUBLAS_DIAG_UNIT>,
	util::enum_container<diagonal_type,
			diagonal_type::non_unit,
			diagonal_type::unit>>;

}
