#pragma once
#include <array>

namespace util {

template <typename type, std::size_t capacity>
struct cyclic_buffer {
/* Tracks up to `capacity` instances of `type`. */
private:
	std::size_t offset;
	std::ptrdiff_t length;
	type buffer[capacity];

	constexpr std::size_t
	index(std::ptrdiff_t i) const
	{
		return (offset + i) % capacity;
	}

	void inc() { length < capacity ? ++length : offset = index(1); }
public:
	template <typename it_type>
	struct iterator {
	private:
		std::size_t index;
		it_type * buffer;

		void inc() { ++index; }
	public:
		it_type& operator*() const { return buffer[index % capacity]; };
		iterator& operator++() { inc(); return *this; };
		iterator operator++(int) { iterator cpy = *this; inc(); return cpy; }
		bool operator==(const iterator& o) { return buffer + index == o.buffer + o.index; }
		bool operator!=(const iterator& o) { return !operator==(o); }

		iterator(std::size_t index, it_type* buffer) :
			index(index), buffer(buffer) {}
	};

	auto begin() { return iterator<type>{index(0), &buffer[0]}; }
	auto end() { return iterator<type>{index(0) + length, &buffer[0]}; }
	auto begin() const { return iterator<const type>{index(0), &buffer[0]}; }
	auto end() const { return iterator<const type>{index(0) + length, &buffer[0]}; }
	std::ptrdiff_t size() const { return length; }

	decltype(auto) operator*() { return *end(); }
	decltype(auto) operator*() const { return *end(); }
	decltype(auto) operator++() { inc(); return *this; }
	decltype(auto) operator++(int) { auto cp = end(); inc(); return cp; }
	constexpr cyclic_buffer() : offset(0), length(0) {}
};

}
