#pragma once
#include "types.h"

namespace cusparse {

enum class direction { row, column };
using direction_adaptor = util::adaptor<
	util::enum_container<direction_t,
			CUSPARSE_DIRECTION_ROW,
			CUSPARSE_DIRECTION_COLUMN>,
	util::enum_container<direction,
			direction::row,
			direction::column>>;

}
