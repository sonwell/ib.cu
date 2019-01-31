#pragma once
#include "types.h"

namespace cusparse {
	enum class action : std::underlying_type_t<action_t> {
		symbolic = CUSPARSE_ACTION_SYMBOLIC,
		numeric = CUSPARSE_ACTION_NUMERIC
	};
}
