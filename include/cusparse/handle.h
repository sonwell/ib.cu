#pragma once
#include <string>
#include <sstream>
#include <cusparse.h>
#include "types.h"
#include "exceptions.h"

namespace cusparse {
	class handle : public type_wrapper<handle_t> {
	protected:
		using type_wrapper<handle_t>::data;
	public:
		handle() { throw_if_error(cusparseCreate(&data)); }
		~handle() { throw_if_error(cusparseDestroy(data)); }
	};
}
