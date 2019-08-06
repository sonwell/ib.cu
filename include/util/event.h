#pragma once

namespace util {

struct event {
	cudaEvent_t _event;

	void record() { cudaEventRecord(_event); }
	event() { cudaEventCreate(&_event); }
	
};

float operator-(event& end, event& start) {
	float ms;

	cudaEventSynchronize(end._event);
	cudaEventElapsedTime(&ms, start._event, end._event);
	return ms;
}

}

