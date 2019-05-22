#pragma once
#include "util/adaptor.h"
#include "cuda/stream.h"
#include "cuda/device.h"
#include "cuda/context.h"
#include "types.h"
#include "exceptions.h"
#include "atomics_mode.h"
#include "math_mode.h"

namespace cublas {

inline void
create(handle_t& handle)
{
	throw_if_error(cublasCreate(&handle),
			"could not create cublas handle");
}

inline void
destroy(handle_t& handle)
{
	throw_if_error(cublasDestroy(handle));
}

class handle : public cuda::context, public cublas::type_wrapper<handle_t> {
protected:
	using context = cuda::context;
	using base = cublas::type_wrapper<handle_t>;
	using base::value;
private:
	using am_a = cublas::atomics_adaptor;
	using mm_a = cublas::math_mode_adaptor;

	auto get_am() const { return get_atomics_mode(*this); }
	void set_am(const am_a& m) { set_atomics_mode(*this, m); }
	auto get_mm() const { return get_math_mode(*this); }
	void set_mm(const mm_a& m) { set_math_mode(*this, m); }
public:
	util::getset<am_a> atomics_mode;
	util::getset<mm_a> math_mode;

	handle(cuda::device& dev = cuda::get_device(), cuda::stream str = cuda::default_stream()) :
		context(dev, str), base(),
		atomics_mode([&] () { return get_am(); }, [&] (const am_a& m) { set_am(m); }),
		math_mode([&] () { return get_mm(); }, [&] (const mm_a& m) { set_mm(m); }) {}
};

}
