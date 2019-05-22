#pragma once
#include "cuda/stream.h"
#include "cuda/device.h"
#include "cuda/context.h"
#include "types.h"
#include "exceptions.h"

namespace cusolver {
namespace dense {

struct adl {};
template <typename value_type, typename lookup = adl>
using type_wrapper = cuda::type_wrapper<value_type, lookup>;

inline void
create(handle_t& handle)
{
	throw_if_error(cusolverDnCreate(&handle));
}

inline void
destroy(handle_t& handle)
{
	throw_if_error(cusolverDnDestroy(handle));
}

class handle : public cuda::context, public dense::type_wrapper<handle_t> {
protected:
	using context = cuda::context;
	using base = dense::type_wrapper<handle_t>;
public:
	handle(cuda::device& dev = cuda::get_device(),
			cuda::stream str = cuda::default_stream()) :
		context(dev, str), base() {}
};

} // namespace dense

namespace sparse {

struct adl {};
template <typename value_type, typename lookup = adl>
using type_wrapper = cuda::type_wrapper<value_type, lookup>;

inline void
create(handle_t& handle)
{
	throw_if_error(cusolverSpCreate(&handle));
}

inline void
destroy(handle_t& handle)
{
	throw_if_error(cusolverSpDestroy(handle));
}

class handle : public cuda::context, public sparse::type_wrapper<handle_t> {
protected:
	using context = cuda::context;
	using base = sparse::type_wrapper<handle_t>;
public:
	handle(cuda::device& dev = cuda::get_device(),
			cuda::stream str = cuda::default_stream()) :
		context(dev, str), base() {}
};

} // namespace sparse
} // namepsace cusolver
