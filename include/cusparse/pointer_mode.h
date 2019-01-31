#pragma once
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusparse {
	enum class pointer_mode : std::underlying_type_t<pointer_mode_t> {
		host = CUSPARSE_POINTER_MODE_HOST,
		device = CUSPARSE_POINTER_MODE_DEVICE
	};

	pointer_mode get_pointer_mode(handle& h)
	{
		pointer_mode_t mode;
		throw_if_error(cusparseGetPointerMode(h, &mode));
		return static_cast<pointer_mode>(mode);
	}

	void set_pointer_mode(handle& h, pointer_mode mode)
	{
		throw_if_error(cusparseSetPointerMode(h, mode));
	}
}
