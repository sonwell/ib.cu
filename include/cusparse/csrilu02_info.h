#pragma once
#include "types.h"

namespace cusparse {
	class csrilu02_info : public type_wrapper<csrilu02_info_t> {
	protected:
		using type_wrapper<csrilu02_info_t>::data;
	public:
		csrilu02_info() { throw_if_error(cusparseCreateCsrilu02Info(&data)); }
		~csrilu02_info() { throw_if_error(cusparseDestroyCsrilu02Info(data)); }
	};
}
