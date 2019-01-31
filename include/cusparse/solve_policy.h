#pragma once
#include "types.h"

namespace cusparse {
	enum class solve_policy : std::underlying_type_t<solve_policy_t> {
		no_level = CUSPARSE_SOLVE_POLICTY_NO_LEVEL,
		use_level = CUSPARSE_SOLVE_POLICY_USE_LEVEL
	};
}
