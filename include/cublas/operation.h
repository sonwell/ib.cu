#pragma once
#include "util/adaptor.h"
#include "cuda/operation.h"
#include "types.h"

namespace cublas {

using cuda::operation;
using operation_adaptor = util::adaptor<
	util::enum_container<operation_t,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			CUBLAS_OP_C>,
	util::enum_container<operation,
			operation::non_transpose,
			operation::transpose,
			operation::conjugate_transpose>>;

}
