#pragma once
#include "cuda/stream.h"
#include "cuda/device.h"
#include "cuda/context.h"
#include "types.h"
#include "exceptions.h"

namespace cusparse {

inline void
create(handle_t& handle)
{
	throw_if_error(cusparseCreate(&handle));
}

inline void
destroy(handle_t& handle)
{
	throw_if_error(cusparseDestroy(handle));
}

class handle : public cuda::context, public cusparse::type_wrapper<handle_t> {
protected:
	using context = cuda::context;
	using base = cusparse::type_wrapper<handle_t>;
public:
	handle(cuda::device& dev = cuda::get_device(), cuda::stream str = cuda::default_stream()) :
		context(dev, str), base() {}
};

}
