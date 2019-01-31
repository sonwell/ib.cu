#pragma once
#include "types.h"

namespace cusparse {
	class csric02_info : public type_wrapper<csric02_info_t> {
	protected:
		using type_wrapper<csric02_info_t>::data;
	public:
		csric02_info() { throw_if_error(cusparseCreateCsric02Info(&data)); }
		~csric02_info() { throw_if_error(cusparseDestroyCsric02Info(data)); }
	};
}
