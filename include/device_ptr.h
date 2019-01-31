#pragma once
#include "util/memory.h"
#include "cuda/device.h"

namespace mem {
	template <typename type>
	struct device_ptr : util::memory<type> {
	private:
		using base_type = util::memory<type>;
	public:
		using typename base_type::allocator_type;

		device_ptr& operator=(device_ptr&& o) { base_type::operator=(std::move(o)); return *this; }

		device_ptr(std::size_t size = 0, allocator_type allocator = default_allocator) :
			base_type(size, allocator) {}
		device_ptr(std::nullptr_t, allocator_type allocator = default_allocator) :
			base_type(nullptr, allocator) {}
		device_ptr(device_ptr&& o) :
			base_type(std::move(o)) {}
		device_ptr(const device_ptr&) = delete;
	private:
		static allocator_type default_allocator;
	};

	template <typename type>
	typename device_ptr<type>::allocator_type
	device_ptr<type>::default_allocator(cuda::default_device().memory());
}
