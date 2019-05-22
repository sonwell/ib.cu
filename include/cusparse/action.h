#pragma once
#include "util/adaptor.h"
#include "types.h"

namespace cusparse {

enum class action { symbolic, numeric };
using action_adaptor = util::adaptor<
	util::enum_container<action_t,
			CUSPARSE_ACTION_SYMBOLIC,
			CUSPARSE_ACTION_NUMERIC>,
	util::enum_container<action,
			action::symbolic,
			action::numeric>>;

}
