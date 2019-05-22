#pragma once
#include "util/allocator.h"
#include "util/memory_resource.h"
#include "exceptions.h"
#include "device.h"

namespace cuda {
	template <typename type>
	class allocator : public util::allocator<type> {
	public:
		virtual type*
		allocate(std::size_t bytes)
		{
			scope lock(dev.id);
			return util::allocator<type>::allocate(bytes);
		}

		virtual void
		deallocate(type* ptr, std::size_t bytes)
		{
			scope lock(dev.id);
			util::allocator<type>::deallocate(ptr, bytes);
		}
	protected:
		device& dev;
	public:
		allocator(device& dev = default_device()) :
			util::allocator<type>(dev.memory()), dev(dev) {}
	};
}
