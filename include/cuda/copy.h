#pragma once
#include <cstdlib>
#include "types.h"

namespace cuda {

template <typename type>
void
dtoh(type* dst, const type* src, std::size_t count)
{
	if (!count) return;
	cudaMemcpy(dst, src, sizeof(type) * count,
			cudaMemcpyDeviceToHost);
}

template <typename type>
void
dtod(type* dst, const type* src, std::size_t count)
{
	if (!count) return;
	cudaMemcpy(dst, src, sizeof(type) * count,
			cudaMemcpyDeviceToDevice);
}

template <typename type>
void
htod(type* dst, const type* src, std::size_t count)
{
	if (!count) return;
	cudaMemcpy(dst, src, sizeof(type) * count,
			cudaMemcpyHostToDevice);
}

template <typename type>
void
htoh(type* dst, const type* src, std::size_t count)
{
	if (!count) return;
	cudaMemcpy(dst, src, sizeof(type) * count,
			cudaMemcpyHostToHost);
}

} // namespace cuda
