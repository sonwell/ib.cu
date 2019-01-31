#pragma once
#include "types.h"
#include "exceptions.h"

namespace cuda {
	class event {
	private:
		event_t evt;
	public:
		static constexpr unsigned int blocking_sync = cudaEventBlockingSync;
		static constexpr unsigned int disable_timing = cudaEventDisableTiming;
		static constexpr unsigned int interprocess = cudaEventInterprocess;

		operator const event_t&() const { return evt; }
		operator event_t() const { return evt; }

		void record(stream_t st = 0) { throw_if_error(cudaEventRecord(evt, st)); }
		void synchronize() const { throw_if_error(cudaEventSynchronize(evt)); }
		bool query() const
		{
			status_t status = cudaEventQuery(evt);
			if (status == cudaSuccess)
				return true;
			else if (status == cudaErrorNotReady)
				return false;
			throw_if_error(status);
			return false;
		}

		event(unsigned int flags = 0) { throw_if_error(cudaEventCreateWithFlags(&evt, flags)); }
		~event() { throw_if_error(cudaEventDestroy(evt)); }
	};

	inline void synchronize(const event& evt) { evt.synchronize(); }

	inline float operator-(const event& end, const event& start)
	{
		float ms;
		synchronize(end);
		throw_if_error(cudaEventElapsedTime(&ms, (const event_t&) start, (const event_t&) end));
		return ms;
	}
}
