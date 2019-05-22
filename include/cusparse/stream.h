#pragma once
#include "cuda/stream.h"
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cusparse {

using cuda::stream;

stream
get_stream(handle& h)
{
	stream_t s;
	throw_if_error(cusparseGetStream(h, &s));
	return stream(s);
}

void
set_stream(handle& h, stream& s)
{
	throw_if_error(cusparseSetStream(h, s));
}

} // namespace cusparse
