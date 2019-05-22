#pragma once
#include "types.h"
#include "handle.h"

namespace cusparse {

inline void
gtsv(handle& h, int m, int n, const float* dl, const float* d, const float* du,
		float* b, int ldb)
{
	throw_if_error(cusparseSgtsv(h, m, n, dl, d, du, b, ldb));
}

inline void
gtsv(handle& h, int m, int n, const double* dl, const double* d,
		const double* du, double* b, int ldb)
{
	throw_if_error(cusparseDgtsv(h, m, n, dl, d, du, b, ldb));
}

inline void
gtsv2_buffer_size_ext(handle& h, int m, int n, const float* dl, const float* d,
		const float* du, const float* b, int ldb, size_t* buffer_size)
{
	throw_if_error(cusparseSgtsv2_bufferSizeExt(h, m, n, dl, d, du, b, ldb,
		buffer_size));
}

inline void
gtsv2_buffer_size_ext(handle& h, int m, int n, const double* dl,
		const double* d, const double* du, const double* b, int ldb,
		size_t* buffer_size)
{
	throw_if_error(cusparseDgtsv2_bufferSizeExt(h, m, n, dl, d, du, b, ldb,
		buffer_size));
}

inline void
gtsv2(handle& h, int m, int n, const float* dl, const float* d, const float* du,
		float* b, int ldb, void* buffer)
{
	throw_if_error(cusparseSgtsv2(h, m, n, dl, d, du, b, ldb, buffer));
}

inline void
gtsv2(handle& h, int m, int n, const double* dl, const double* d,
		const double* du, double* b, int ldb, void* buffer)
{
	throw_if_error(cusparseDgtsv2(h, m, n, dl, d, du, b, ldb, buffer));
}

inline void
gtsv_nopivot(handle& h, int m, int n, const float* dl, const float* d,
		const float* du, float* b, int ldb)
{
	throw_if_error(cusparseSgtsv_nopivot(h, m, n, dl, d, du, b, ldb));
}

inline void
gtsv_nopivot(handle& h, int m, int n, const double* dl, const double* d,
		const double* du, double* b, int ldb)
{
	throw_if_error(cusparseDgtsv_nopivot(h, m, n, dl, d, du, b, ldb));
}

inline void
gtsv2_nopivot_buffer_size_ext(handle& h, int m, int n, const float* dl,
		const float* d, const float* du, const float* b, int ldb,
		size_t* buffer_size)
{
	throw_if_error(cusparseSgtsv2_nopivot_bufferSizeExt(h, m, n, dl, d, du,
		b, ldb, buffer_size));
}

inline void
gtsv2_nopivot_buffer_size_ext(handle& h, int m, int n, const double* dl,
		const double* d, const double* du, const double* b, int ldb,
		size_t* buffer_size)
{
	throw_if_error(cusparseDgtsv2_nopivot_bufferSizeExt(h, m, n, dl, d, du,
		b, ldb, buffer_size));
}

inline void
gtsv2_nopivot(handle& h, int m, int n, const float* dl, const float* d,
		const float* du, float* b, int ldb, void* buffer)
{
	throw_if_error(cusparseSgtsv2_nopivot(h, m, n, dl, d, du, b, ldb,
		buffer));
}

inline void
gtsv2_nopivot(handle& h, int m, int n, const double* dl, const double* d,
		const double* du, double* b, int ldb, void* buffer)
{
	throw_if_error(cusparseDgtsv2_nopivot(h, m, n, dl, d, du, b, ldb,
		buffer));
}

inline void
gtsv_strided_batch(handle& h, int m, const float* dl, const float* d,
		const float* du, float* x, int batch_count, int batch_stride)
{
	throw_if_error(cusparseSgtsvStridedBatch(h, m, dl, d, du, x,
		batch_count, batch_stride));
}

inline void
gtsv_strided_batch(handle& h, int m, const double* dl, const double* d,
		const double* du, double* x, int batch_count, int batch_stride)
{
	throw_if_error(cusparseDgtsvStridedBatch(h, m, dl, d, du, x,
		batch_count, batch_stride));
}

inline void
gtsv2_strided_batch_buffer_size_ext(handle& h, int m, const float* dl,
		const float* d, const float* du, const float* x, int batch_count,
		int batch_stride, size_t* buffer_size)
{
	throw_if_error(cusparseSgtsv2StridedBatch_bufferSizeExt(h, m, dl, d, du,
		x, batch_count, batch_stride, buffer_size));
}

inline void
gtsv2_strided_batch_buffer_size_ext(handle& h, int m, const double* dl,
		const double* d, const double* du, const double* x, int batch_count,
		int batch_stride, size_t* buffer_size)
{
	throw_if_error(cusparseDgtsv2StridedBatch_bufferSizeExt(h, m, dl, d, du,
		x, batch_count, batch_stride, buffer_size));
}

inline void
gtsv2_strided_batch(handle& h, int m, const float* dl, const float* d,
		const float* du, float* x, int batch_count, int batch_stride,
		void* buffer)
{
	throw_if_error(cusparseSgtsv2StridedBatch(h, m, dl, d, du, x,
		batch_count, batch_stride, buffer));
}

inline void
gtsv2_strided_batch(handle& h, int m, const double* dl, const double* d,
		const double* du, double* x, int batch_count, int batch_stride,
		void* buffer)
{
	throw_if_error(cusparseDgtsv2StridedBatch(h, m, dl, d, du, x,
		batch_count, batch_stride, buffer));
}

inline void
gtsv_interleaved_batch_buffer_size_ext(handle& h, int algo, int m,
		const float* dl, const float* d, const float* du, const float* x,
		int batch_count, size_t* buffer_size)
{
	throw_if_error(cusparseSgtsvInterleavedBatch_bufferSizeExt(h, algo, m,
		dl, d, du, x, batch_count, buffer_size));
}

inline void
gtsv_interleaved_batch_buffer_size_ext(handle& h, int algo, int m,
		const double* dl, const double* d, const double* du, const double* x,
		int batch_count, size_t* buffer_size)
{
	throw_if_error(cusparseDgtsvInterleavedBatch_bufferSizeExt(h, algo, m,
		dl, d, du, x, batch_count, buffer_size));
}

inline void
gtsv_interleaved_batch(handle& h, int algo, int m, float* dl, float* d,
		float* du, float* x, int batch_count, void* buffer)
{
	throw_if_error(cusparseSgtsvInterleavedBatch(h, algo, m, dl, d, du, x,
		batch_count, buffer));
}

inline void
gtsv_interleaved_batch(handle& h, int algo, int m, double* dl, double* d,
		double* du, double* x, int batch_count, void* buffer)
{
	throw_if_error(cusparseDgtsvInterleavedBatch(h, algo, m, dl, d, du, x,
		batch_count, buffer));
}

} // namespace cusparse