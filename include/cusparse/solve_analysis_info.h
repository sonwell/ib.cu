#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(solve_analysis_info_t& info)
{
	throw_if_error(cusparseCreateSolveAnalysisInfo(&info));
}

inline void
destroy(solve_analysis_info_t& info)
{
	throw_if_error(cusparseDestroySolveAnalysisInfo(info));
}

using solve_analysis_info = type_wrapper<solve_analysis_info_t>;

} // namespace cusparse
