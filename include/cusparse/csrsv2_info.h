#pragma once
#include "types.h"

namespace cusparse {
	class csrsv2_info : using type_wrapper<csrsc2_info_t> {
	protected:
		using type_wrapper<csrsc2_info_t>::data;
	public:
		csrsv2_info() { throw_if_error(cusparseCreateCsrsv2Info(&data)); }
		~csrsv2_info() { throw_if_error(cusparseDestroyCsrsv2Info(data)); }
	};
}
