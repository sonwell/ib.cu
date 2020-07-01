#pragma once

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
void
launch(func_t&& f, int num_ctas, arg_t&& ... args)
{
	if (num_ctas > 0)
		landing<nt, vt><<<num_ctas, nt>>>(std::forward<func_t>(f),
			std::forward<arg_t>(args)...);
}

template <int nt=128, int vt=1, typename func_t, typename ... arg_t>
void
transform(func_t&& f, int count, arg_t&& ... args)
{
	enum { nv = nt * vt };
	int num_ctas = (count + nv - 1) / nv;

	auto k = [=] __device__ (int tid, int cta)
	{
		for (int i = 0; i < vt; ++i)
			if (nv * cta + tid + nt * i < count)
				f(nv * cta + tid + nt * i, args...);
	};

	launch<nt, vt>(std::move(k), num_ctas);
}

}
