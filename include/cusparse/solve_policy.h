#pragma once
#include "util/adaptor.h"
#include "types.h"

namespace cusparse {

enum class solve_policy { no_level, use_level };
using solve_policy_adaptor = util::adaptor<
	util::enum_container<solve_policy_t,
			CUSPARSE_SOLVE_POLICY_NO_LEVEL,
			CUSPARSE_SOLVE_POLICY_USE_LEVEL>,
	util::enum_contaner<solve_policy,
			solve_policy::no_level,
			solve_policy::use_level>>;

}
