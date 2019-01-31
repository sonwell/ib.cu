#pragma once
#include "types.h"

namespace cusparse {
	class csrgemm2_info : public type_wrapper<csrgemm2_info_t> {
	protected:
		using type_wrapper<csrgemm2_info_t>::data;
	public:
		csrgemm2_info() { throw_if_error(cusparseCreateCsrgemm2Info(&data)); }
		~csrgemm2_info() { throw_if_error(cusparseDestroyCsrgemm2Info(data)); }
	};
}
