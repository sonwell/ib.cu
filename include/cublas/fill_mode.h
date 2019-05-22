#pragma once
#include "cuda/fill_mode.h"
#include "util/adaptor.h"
#include "types.h"

namespace cublas {

using cuda::fill_mode;
using fill_mode_adaptor = util::adaptor<
	util::enum_container<fill_mode_t,
			CUBLAS_FILL_MODE_LOWER,
			CUBLAS_FILL_MODE_UPPER>,
	util::enum_container<fill_mode,
			fill_mode::lower,
			fill_mode::upper>>;

} // namespace cublas
