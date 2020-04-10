#pragma once
#include "cuda/exceptions.h"
#include "cuda/device.h"

namespace util {

template <int nt, int, typename func_t, typename ... arg_t>
__global__ void __launch_bounds__(nt, 0)
landing(func_t f, arg_t ... args)
{
	int tid = threadIdx.x % nt;
	int cta = blockIdx.x;

	f(tid, cta, std::move(args)...);
}

template <int nt, int vt, typename func_t, typename ... arg_t>
__host__ void
launch(func_t&& f, int num_ctas, arg_t&& ... args)
{
	if (num_ctas > 0)
		landing<nt, vt><<<num_ctas, nt>>>(std::forward<func_t>(f),
			std::forward<arg_t>(args)...);
}

template <int nt=128, int vt=1, typename func_t, typename ... arg_t>
__host__ void
transform(func_t&& f, int count, arg_t&& ... args)
{
	enum { nv = nt * vt };
	int num_ctas = (count + nv - 1) / nv;

	auto k = [=] __device__ (int tid, int cta)
	{
		for (int i = 0; i < vt; ++i)
			if (nv * cta + tid + nt * i < count)
				f(nv * cta + tid + nt * i, std::move(args)...);
	};

	launch<nt, vt>(std::move(k), num_ctas);
}

template <int max_threads=1, int nt=128, int vt=1, typename func_t, typename ... arg_t>
__host__ void
strong_transform(func_t&& f, int count, arg_t&& ... args)
{
	constexpr int mt = max_threads < nt ? max_threads : nt;
	constexpr int mv = max_threads * vt;
	int iterations = (count + mv - 1) / mv;
	int items_per_block = iterations * mt * vt;
	int num_ctas = (count + items_per_block - 1) / items_per_block;

	auto k = [=] __device__ (int tid, int cta)
	{
		for (int j = 0; j < iterations; ++j)
			for (int i = 0; i < vt; ++i) {
				int id = cta * items_per_block + mt * vt * j + mt * i + tid;
				if (id < count && tid < max_threads)
					f(id, args...);
			}
	};

	launch<nt, vt>(std::move(k), num_ctas);
}

}
