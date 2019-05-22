#pragma once
#include "cuda/pointer.h"
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusparse {

using cuda::pointer_mode;
using pointer_mode_adaptor = util::adaptor<
	util::enum_container<pointer_mode_t,
			CUSPARSE_POINTER_MODE_HOST,
			CUSPARSE_POINTER_MODE_DEVICE>,
	util::enum_container<pointer_mode,
			pointer_mode::host,
			pointer_mode::device>>;

inline pointer_mode
get_pointer_mode(handle& h)
{
	pointer_mode_t mode;
	throw_if_error(cusparseGetPointerMode(h, &mode));
	return pointer_mode_adaptor(mode);
}

inline void
set_pointer_mode(handle& h, pointer_mode_adaptor mode)
{
	throw_if_error(cusparseSetPointerMode(h, mode));
}

}
