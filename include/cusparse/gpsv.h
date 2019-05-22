#pragma once
#include "types.h"
#include "handle.h"

namespace cusparse {

inline void
gpsv_interleaved_batch_buffer_size_ext(handle& h, int algo, int m,
		const float* ds, const float* dl, const float* d, const float* du,
		const float* dw, const float* x, int batch_count, size_t* buffer_size)
{
	throw_if_error(cusparseSgpsvInterleavedBatch_bufferSizeExt(h, algo, m,
		ds, dl, d, du, dw, x, batch_count, buffer_size));
}

inline void
gpsv_interleaved_batch_buffer_size_ext(handle& h, int algo, int m,
		const double* ds, const double* dl, const double* d, const double* du,
		const double* dw, const double* x, int batch_count, size_t* buffer_size)
{
	throw_if_error(cusparseDgpsvInterleavedBatch_bufferSizeExt(h, algo, m,
		ds, dl, d, du, dw, x, batch_count, buffer_size));
}

inline void
gpsv_interleaved_batch(handle& h, int algo, int m, float* ds, float* dl,
		float* d, float* du, float* dw, float* x, int batch_count, void* buffer)
{
	throw_if_error(cusparseSgpsvInterleavedBatch(h, algo, m, ds, dl, d, du,
		dw, x, batch_count, buffer));
}

inline void
gpsv_interleaved_batch(handle& h, int algo, int m, double* ds, double* dl,
		double* d, double* du, double* dw, double* x, int batch_count,
		void* buffer)
{
	throw_if_error(cusparseDgpsvInterleavedBatch(h, algo, m, ds, dl, d, du,
		dw, x, batch_count, buffer));
}

} // namespace cusparse