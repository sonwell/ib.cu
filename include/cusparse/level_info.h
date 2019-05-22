#pragma once
#include "handle.h"
#include "solve_analysis_info.h"

namespace cusparse {

struct level_info {
	int nlevels;
	int* levels_ptr;
	int* levels_ind;
};

inline void
get_level_info(handle& h, solve_analysis_info& info, level_info& li)
{
	throw_if_error(cusparseGetLevelInfo(h, info, &(li.nlevels),
			&(li.levels_ptr), &(li.levels_ind)));
}

}
