#pragma once
#include "types.h"

namespace cusparse {
	enum class direction : std::underlying_type_t<direction_t> {
		row = CUSPARSE_DIRECTION_ROW,
		column = CUSPARSE_DIRECTION_COLUMN
	};
}
