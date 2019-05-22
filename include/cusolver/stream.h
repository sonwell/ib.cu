#pragma once
#include "cuda/stream.h"
#include "types.h"
#include "exceptions.h"

namespace cusolver {

using cuda::stream;

stream
get_stream(dense::handle_t h)
{
	stream_t s;
	throw_if_error(cusolverDnGetStream(h, &s));
	return stream(s);
}

stream
get_stream(sparse::handle_t h)
{
	stream_t s;
	throw_if_error(cusolverSpGetStream(h, &s));
	return stream(s);
}

void
set_stream(dense::handle_t h, stream_t s)
{
	throw_if_error(cusolverDnSetStream(h, s));
}

void
set_stream(sparse::handle_t h, stream_t s)
{
	throw_if_error(cusolverSpSetStream(h, s));
}

} // namespace cusolver
