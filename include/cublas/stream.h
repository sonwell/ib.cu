#pragma once
#include "cuda/stream.h"
#include "types.h"
#include "exceptions.h"
#include "handle.h"

namespace cublas {

using cuda::stream;

stream
get_stream(handle_t h)
{
	stream_t s;
	throw_if_error(cublasGetStream(h, &s));
	return stream(s);
}

void
set_stream(handle_t h, stream& s)
{
	throw_if_error(cublasSetStream(h, s));
}

} // namespace cublas
