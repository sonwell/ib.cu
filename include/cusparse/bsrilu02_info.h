#pragma once
#include "types.h"

namespace cusparse {
	class bsrilu02_info : public type_wrapper<bsrilu02_info_t> {
	protected:
		using type_wrapper<bsrilu02_info_t>::data;
	public:
		bsrilu02_info() { throw_if_error(cusparseCreateBsrilu02Info(&data)); }
		~bsrilu02_info() { throw_if_error(cusparseDestroyBsrilu02Info(data)); }
	};
}
