#pragma once

#include "util/adaptor.h"
#include "types.h"

namespace cublas {

enum class side_mode { left, right };

using side_mode_adaptor = util::adaptor<
	util::enum_container<side_mode_t,
			CUBLAS_SIDE_LEFT,
			CUBLAS_SIDE_RIGHT>,
	util::enum_container<side_mode,
			side_mode::left,
			side_mode::right>>;

} // namespace cublas
