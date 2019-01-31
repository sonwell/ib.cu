#pragma once
#include "types.h"

namespace cusparse {
	class bsrsm2_info : public type_wrapper<bsrsm2_info_t> {
	protected:
		using type_wrapper<bsrsm2_info_t>::data;
	public:
		bsrsm2_info() { throw_if_error(cusparseCreateBsrsm2Info(&data)); }
		~bsrsm2_info() { throw_if_error(cusparseDestroyBsrsm2Info(data)); }
	};
}
