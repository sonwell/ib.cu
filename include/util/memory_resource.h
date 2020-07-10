#pragma once
#include <cstddef>
#include <memory>
#include <new>

namespace util {

// This is basically std::pmr::memory_resource
struct memory_resource /* , std::pmr::memory_resource */ {
private:
	virtual void* do_allocate(std::size_t, std::size_t) = 0;
	virtual void do_deallocate(void*, std::size_t, std::size_t) = 0;

	virtual bool
	do_is_equal(const memory_resource& other) const noexcept
	{
		return this == &other;
	}
public:
	void*
	allocate(std::size_t bytes,
			std::size_t alignment = alignof(std::max_align_t))
	{
		return do_allocate(bytes, alignment);
	}

	void
	deallocate(void* ptr, std::size_t bytes,
			std::size_t alignment = alignof(std::max_align_t))
	{
		return do_deallocate(ptr, bytes, alignment);
	}

	bool
	is_equal(const memory_resource& other) const noexcept
	{
		return do_is_equal(other);
	}

	memory_resource() {};
	virtual ~memory_resource() = default;
};

inline bool
operator==(const memory_resource& a, const memory_resource& b)
{
	return a.is_equal(b);
}

inline bool
operator!=(const memory_resource& a, const memory_resource& b)
{
	return !a.is_equal(b);
}

class main_memory_resource : public memory_resource {
private:
	virtual void*
	do_allocate(std::size_t bytes, std::size_t alignment)
	{
		auto align_val = static_cast<std::align_val_t>(alignment);
		return ::operator new(bytes, align_val);
	}

	virtual void
	do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
	{
		auto align_val = static_cast<std::align_val_t>(alignment);
		// clang: -fsized-deallocate
		return ::operator delete(ptr, bytes, align_val);
	}
public:
	using memory_resource::memory_resource;
};

inline memory_resource*
new_delete_resource() noexcept
{
	static main_memory_resource main_memory;
	return &main_memory;
}

// Pre-allocates a huge block of memory to use, then doles it out in the chunks
// of the requested size. Could be used to speed up CUDA memory allocations.
class static_memory_resource : public memory_resource {
private:
	class node {
		std::ptrdiff_t start;
		std::ptrdiff_t end;
		node* next;

		node(std::ptrdiff_t start, std::ptrdiff_t end) :
			start(start), end(end), next(nullptr) {}
		node(std::ptrdiff_t start, std::ptrdiff_t end, node* next) :
			start(start), end(end), next(next) {}
		~node() { if (next != nullptr) delete next; }

		friend class static_memory_resource;
	};

	memory_resource* parent_resource;
	std::size_t bytes;
	void* memory;
	node* head;

	virtual bool
	do_is_equal(const memory_resource& other) const noexcept
	{
		const static_memory_resource* smr = dynamic_cast<const static_memory_resource*>(&other);
		if (!smr) return false;
		return *parent_resource == *(smr->parent_resource) &&
			memory == smr->memory;
	}

	virtual void*
	do_allocate(std::size_t bytes, std::size_t alignment)
	{
		if (!bytes) return nullptr;
		auto* m_ptr = static_cast<std::byte*>(memory);

		void* tmp = nullptr;
		std::size_t size;
		auto can_fit = [&] (node* n) {
			tmp = static_cast<void*>(m_ptr + n->start);
			size = n->end - n->start;
			std::align(alignment, bytes, tmp, size);
			return tmp != nullptr && size > 0;
		};

		node* curr = head;
		while (curr != nullptr && !can_fit(curr))
			curr = curr->next;
		if (curr == nullptr)
			return parent_resource->allocate(bytes, alignment);

		std::ptrdiff_t diff = static_cast<std::byte*>(tmp) - (m_ptr + curr->start);
		node* mark = new node(curr->start + diff + bytes, curr->end, curr->next);
		curr->end = curr->start + diff;
		curr->next = mark;
		return tmp;
	}

	virtual void
	do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
	{
		if (ptr == nullptr) return;
		auto* m_ptr = static_cast<std::byte*>(memory);
		auto* p_ptr = static_cast<std::byte*>(ptr);
		std::ptrdiff_t diff = p_ptr - m_ptr;
		if (diff < 0 || diff > size())
			return parent_resource->deallocate(ptr, bytes, alignment);

		node* curr = head;
		node* last = nullptr;
		while (curr != nullptr && curr->end <= diff) {
			last = curr;
			curr = last->next;
		}
		if (curr == nullptr) return;

		if (last->end == diff)
			last->end = last->end + bytes;
		else
			last = (last->next = new node(diff, diff+bytes, curr));
		if (last->end == curr->start) {
			last->end = curr->end;
			last->next = curr->next;
			curr->next = nullptr;
			delete curr;
		}
	}
public:
	std::size_t size() const { return bytes; }
	memory_resource* resource() const { return parent_resource; }

	static_memory_resource(memory_resource* resource, std::size_t bytes) :
		parent_resource(resource), bytes(bytes),
		memory(resource->allocate(bytes)),
		head(new node(0, bytes)) {}
	static_memory_resource(const static_memory_resource& other) = delete;

	virtual ~static_memory_resource()
	{
		parent_resource->deallocate(memory, bytes);
		delete head;
	}
};

namespace detail {

inline memory_resource*&
default_resource() noexcept
{
	static memory_resource* default_resource = new_delete_resource();
	static thread_local memory_resource* thread_resource = default_resource;
	return thread_resource;
}

} // namespace detail

inline memory_resource*
get_default_resource() noexcept
{
	return detail::default_resource();
}

inline memory_resource*
set_default_resource(memory_resource* r) noexcept
{
	return detail::default_resource() = r;
}

} // namespace util
