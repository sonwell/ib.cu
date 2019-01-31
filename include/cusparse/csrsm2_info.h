#pragma once
#include "types.h"

namespace cusparse {
	class csrsm2_info : public type_wrapper<csrsm2_info_t> {
	protected:
		using type_wrapper<csrsm2_info_t>::data;
	public:
		csrsm2_info() { throw_if_error(cusparseCreateCsrsm2Info(&data)); }
		~csrsm2_info() { throw_if_error(cusparseDestroyCsrsm2Info(data)); }
	};
}
