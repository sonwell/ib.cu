#pragma once
#include <cstddef>
#include "allocator.h"

namespace util {
	template <typename type>
	class memory {
	protected:
		void
		swap(memory& o)
		{
			std::swap(_alloc, o._alloc);
			std::swap(_size, o._size);
			std::swap(_ptr, o._ptr);
		}
	public:
		using value_type = type;
		using allocator_type = allocator<value_type>;
		using pointer = value_type*;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		//using reference = value_type&;
		//using const_reference = const value_type&;

		allocator_type get_allocator() const { return _alloc; }

		operator pointer() const { return _ptr; }
		pointer data() const { return _ptr; }
		std::size_t size() const { return _size; }
		memory& operator=(memory&& o) { swap(o); return *this; }
		memory& operator=(std::nullptr_t) { this->~memory(); return *this; }

		memory(std::size_t size = 0, const allocator_type& alloc = allocator_type()) :
			_alloc(alloc), _size(size), _ptr(size ? _alloc.allocate(_size) : nullptr) {}
		memory(std::nullptr_t, const allocator_type& alloc = allocator_type()) :
			_alloc(alloc), _size(0), _ptr(nullptr) {}
		memory(memory&& o) : _size(0), _ptr(nullptr) { swap(o); }
		memory(const memory&) = delete;
		~memory()
		{
			_alloc.deallocate(_ptr, _size);
			_ptr = nullptr;
			_size = 0;
		}
	private:
		allocator_type _alloc;
		size_type _size;
		pointer _ptr;
	};

	template <typename type> memory(std::size_t, const allocator<type>&) -> memory<type>;
	template <typename type> memory(std::nullptr_t, const allocator<type>&) -> memory<type>;
	memory(std::size_t) -> memory<void>;
	memory(std::nullptr_t) -> memory<void>;
	memory() -> memory<void>;
}
