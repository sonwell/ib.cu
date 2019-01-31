#pragma once

#include "types.h"

namespace cusparse {
	enum class operation : std::underlying_type_t<operation_t> {
		non_transpose = CUSPARSE_OPERATION_NON_TRANSPOSE,
		transpose = CUSPARSE_OPERATION_TRANSPOSE,
		conjugate_transpose = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
	};
}
