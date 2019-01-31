#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {
	enum class hyb_partition : std::underlying_type_t<hyb_partition_t> {
		automatic = CUSPARSE_HYB_PARTITION_AUTO,
		user = CUSPARSE_HYB_PARTITION_USER,
		maximum = CUSPARSE_HYB_PARTITION_MAX
	};

	class hyb_matrix : public type_wrapper<hyb_mat_t> {
	protected:
		using type_wrapper<hyb_mat_t>::data;
	public:
		hyb_matrix() { throw_if_error(cusparseCreateHybMat(&data)); }
		~hyb_matrix() { throw_if_error(cusparseDestroyHybMat(data)); }
	};
}
