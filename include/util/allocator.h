#pragma once
#include <cstddef>
#include "memory_resource.h"
#include "device.h"

namespace util {
	template <typename type>
	class allocator {
	public:
		using value_type = type;

		virtual type*
		allocate(std::size_t count)
		{
			auto* ptr = memory->allocate(count * sizeof(type), alignof(type));
			return static_cast<type*>(ptr);
		}

		virtual void
		deallocate(type* ptr, std::size_t count)
		{
			return memory->deallocate(static_cast<void*>(ptr),
					count * sizeof(type), alignof(type));
		}

		memory_resource* resource() const { return memory; }

		allocator(memory_resource* memory = get_default_resource()) :
			memory(memory) {}
	private:
		memory_resource* memory;
	};
}
