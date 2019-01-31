#pragma once
#include "util/heap.h"
#include "types.h"
#include "allocator.h"

namespace cuda {
	enum class attach_association : unsigned int {
		global = cudaMemAttachGlobal,
		host= cudaMemAttachHost,
		single = cudaMemAttachSingle
	};

	class stream {
	private:
		stream_t stm;
	public:
		static constexpr unsigned int non_blocking = cudaStreamNonBlocking;
		const unsigned int flags;
		const int priority;

		operator const stream_t&() const { return stm;}
		operator stream_t() { return stm;}

		void synchronize() const { throw_if_error(cudaStreamSynchronize(stm)); }
		void wait(event_t evt) const { throw_if_error(cudaStreamWaitEvent(stm, evt, 0)); }

		template <typename type>
		void attach(util::heap<type>& heap, attach_association assoc)
		{
			void* ptr = heap.data();
			std::size_t size = heap.size() * sizeof(type);
			throw_if_error(cudaStreamAttachMemAsync(srm, ptr, size, (unsigned int) assoc));
		}

		bool query() const
		{
			status_t status = cudaStreamQuery(stm);
			if (status == cudaSuccess)
				return true;
			else if (status == cudaErrorNotReady)
				return false;
			throw_if_error(status);
		}

		stream(unsigned int flags = 0, int priority = 0) :
			flags(flags), priority(priority)
		{
			throw_if_error(cudaStreamCreateWithPriority(&stm, flags, priority));
		}
		~stream() { throw_if_error(cudaStreamDestroy(stm)); }
	};

	inline void synchronize(const stream& stm) { stm.synchronize(); }
};
