#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
#include "util/memory_resource.h"
#include "exceptions.h"
#include "scope.h"

namespace cuda {

class device;

class device_memory : public util::memory_resource {
private:
	const device& dev;

	virtual void*
	do_allocate(std::size_t bytes, std::size_t)
	{
		void* ptr;
		scope lock(dev);
		throw_if_error(cudaMalloc(&ptr, bytes),
				"could not allocate GPU memory");
		return ptr;
	}

	virtual void
	do_deallocate(void* ptr, std::size_t, std::size_t)
	{
		scope lock(dev);
		throw_if_error(cudaFree(ptr),
				"could not free GPU memory");
	}
public:
	device_memory(const device& dev) :
		dev(dev) {}

};
} // namespace cuda
