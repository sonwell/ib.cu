#pragma once
#include "util/adaptor.h"
#include "linalg/operation.h"
#include "types.h"

namespace cusparse {

using linalg::operation;
using operation_adaptor = util::adaptor<
	util::enum_container<operation_t,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_TRANSPOSE,
			CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE>,
	util::enum_container<cuda::operation,
			operation::non_transpose,
			operation::transpose,
			operation::conjugate_transpose>>;

}
