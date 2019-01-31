#pragma once
#include "types.h"

namespace cusparse {
	class bsrsv2_info : public type_wrapper<bsrsv2_info_t>{
	protected:
		using type_wrapper<bsrsv2_info_t>::data;
	public:
		bsrsv2_info() { throw_if_error(cusparseCreateBsrsv2Info(&data)); }
		~bsrsv2_info() { throw_if_error(cusparseDestroyBsrsv2Info(data)); }
	};
}
