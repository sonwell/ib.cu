#pragma once
#include "cuda/stream.h"
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusparse {
	cuda::stream
	get_stream(handle& h)
	{
		cuda::stream_t stream;
		throw_if_error(cusparseGetStream(h, &stream));
		return cuda::stream(stream);
	}

	void
	set_stream(handle& h, cuda::stream& s)
	{
		throw_if_error(cusparseSetStream(h, stream));
	}
};
