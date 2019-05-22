#pragma once
#include <memory>
#include "util/memory.h"
#include "exceptions.h"
#include "types.h"

namespace cuda {

enum class attach_association : unsigned int {
	global = cudaMemAttachGlobal,
	host= cudaMemAttachHost,
	single = cudaMemAttachSingle
};

inline void
create(stream_t& stream, unsigned int flags = 0, int priority = 0)
{
	throw_if_error(cudaStreamCreateWithPriority(&stream, flags, priority),
			"could not construct stream");
}

inline void
destroy(stream_t& stream)
{
	if (stream == nullptr) return;
	throw_if_error(cudaStreamDestroy(stream));
}

class stream;
stream& default_stream();

class stream : public type_wrapper<stream_t> {
protected:
	using base = type_wrapper<stream_t>;
	using base::internal;

	stream(internal, std::nullptr_t) :
		base((stream_t) nullptr), flags(0), priority(0) {}
public:
	static constexpr unsigned int non_blocking = cudaStreamNonBlocking;
	const unsigned int flags;
	const int priority;

	void synchronize() const { throw_if_error(cudaStreamSynchronize(*this)); }
	void wait(event_t evt) const { throw_if_error(cudaStreamWaitEvent(*this, evt, 0)); }

	template <typename type>
	void attach(util::memory<type>& mem, attach_association assoc)
	{
		void* ptr = mem.data();
		std::size_t size = mem.size() * sizeof(type);
		unsigned int ua = static_cast<unsigned int>(assoc);
		throw_if_error(cudaStreamAttachMemAsync(*this, ptr, size, ua));
	}

	bool query() const
	{
		status_t status = cudaStreamQuery(*this);
		if (status == cudaSuccess)
			return true;
		else if (status == cudaErrorNotReady)
			return false;
		throw exception(status);
	}

	stream(unsigned int flags = 0, int priority = 0) :
		base(flags, priority),
		flags(flags), priority(priority) {}
	stream(std::nullptr_t) :
		stream(default_stream()) {}
	stream(stream_t s) :
		base(s), flags(0), priority(0) {}
	stream(const stream& s) :
		base(s), flags(s.flags), priority(s.priority) {}

friend stream& default_stream();
};

stream&
default_stream()
{
	static stream default_stream{stream::internal{}, nullptr};
	return default_stream;
}

inline void synchronize(const stream& stm) { stm.synchronize(); }

} // namespace cuda
