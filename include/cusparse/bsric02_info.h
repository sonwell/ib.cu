#pragma once
#include "types.h"

namespace cusparse {
	class bsric02_info : public type_wrapper<bsric02_info_t> {
	protected:
		using type_wrapper<bsric02_info_t>::data;
	public:
		bsric02_info() { throw_if_error(cusparseCreateBsric02Info(&data)); }
		~bsric02_info() { throw_if_error(cusparseDestroyBsric02Info(&data)); }
	};
}
