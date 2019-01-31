#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
#include "util/memory_resource.h"
#include "exceptions.h"
#include "scope.h"

namespace cuda {
	class device_memory : public util::memory_resource {
	private:
		int id;

		virtual void*
		do_allocate(std::size_t bytes, std::size_t)
		{
			void* ptr;
			scope lock(id);
			throw_if_error(cudaMalloc(&ptr, bytes));
			return ptr;
		}

		virtual void
		do_deallocate(void* ptr, std::size_t, std::size_t)
		{
			scope lock(id);
			throw_if_error(cudaFree(ptr));
		}
	public:
		device_memory(int id = 0) : id(id) {}
	};
}
