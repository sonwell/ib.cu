#pragma once
#include "types.h"
#include "exceptions.h"

namespace cusparse {
	class prune_info : public type_wrapper<prune_info_t> {
	protected:
		using type_wrapper<prune_info_t>::data;
	public:
		prune_info() { throw_if_error(cusparseCreatePruneInfo(&data)); }
		~prune_info() { throw_if_error(cusparseDestroyPruneInfo(data)); }
	};
}
