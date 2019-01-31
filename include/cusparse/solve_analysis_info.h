#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {
	class solve_analysis_info : public type_wrapper<solve_analysis_info_t> {
	protected:
		using type_wrapper<solve_analysis_info_t>::data;
	public:
		solve_analysis_info() { throw_if_error(cusparseCreateSolveAnalysisInfo(&data)); }
		~solve_analysis_info() { throw_if_error(cusparseDestroySolveAnalysisInfo(data)); }
	};
}
