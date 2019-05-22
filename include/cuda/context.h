#pragma once
#include "device.h"
#include "stream.h"

namespace cuda {

class context {
protected:
	cuda::device& dev;
	cuda::stream str;
public:
	const cuda::device& device() const { return dev; }
	cuda::device& device() { return dev; }
	cuda::stream stream() const { return str; }

	context(const context&) = delete;
	context& operator=(const context&) = delete;

	context(cuda::device& dev = get_device(), cuda::stream str = default_stream()) :
		dev(dev), str(str) {}
};

} // namespace cuda
