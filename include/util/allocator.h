#pragma once
#include <type_traits>
#include <cstddef>
#include "memory_resource.h"

namespace util {
namespace detail {

template <typename type>
struct memory_traits {
	static constexpr auto size = sizeof(type);
	static constexpr auto align = alignof(type);
};

template <>
struct memory_traits<void> : memory_traits<std::byte> {};

} // namespace detail

// A pared-down std::pmr::polymorphic_allocator
template <typename type>
class allocator {
public:
	using value_type = type;

	virtual type*
	allocate(std::size_t count)
	{
		auto* ptr = memory->allocate(count * type_size, type_align);
		return static_cast<type*>(ptr);
	}

	virtual void
	deallocate(type* ptr, std::size_t count)
	{
		return memory->deallocate(static_cast<void*>(ptr),
				count * type_size, type_align);
	}

	memory_resource* resource() const { return memory; }

	__host__ allocator(memory_resource* memory = get_default_resource()) :
		memory(memory) {}
private:
	using memory_traits = detail::memory_traits<type>;
	static constexpr std::size_t type_size = memory_traits::size;
	static constexpr std::size_t type_align = memory_traits::align;
	memory_resource* memory;
};

} // namespace util
