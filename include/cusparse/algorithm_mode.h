#pragma once
#include "types.h"

namespace cusparse {
	enum class algorithm_mode : std::underlying_type_t<algorithm_mode_t> {
		naive = CUSPARSE_ALG_NAIVE,
		merge_path = CUSPARSE_ALG_MERGE_PATH
	};
}
