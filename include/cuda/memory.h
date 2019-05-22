#pragma once

namespace cuda {
	template <int nt, int vt, typename type, std::size_t size>
	__device__ void
	mem_to_shared(const type* mem, type (&shared)[size],
			int tid, int cta, int count, bool sync = true)
	{
		for (int i = 0; i < vt; ++i) {
			int j = nt * vt * cta + i * nt + tid;
			if (j < count) shared[i * nt + tid] = mem[j];
		}
		if (sync) __syncthreads();
	}

	template <int nt, int vt, typename type, std::size_t size1, std::size_t size2>
	__device__ void
	shared_to_reg(const type (&shared)[size1], type (&reg)[size2],
			int tid, bool sync = true)
	{
		for (int i = 0; i < vt; ++i)
			reg[i] = shared[vt * tid + i];
		if (sync) __syncthreads();
	}

	template <int nt, int vt, typename type, std::size_t size1, std::size_t size2>
	__device__ void
	mem_to_reg(const type* mem, type (&shared)[size1], type (&reg)[size2],
			int tid, int cta, int count, bool sync = true)
	{
		mem_to_shared<nt, vt>(mem, shared, tid, cta, count);
		shared_to_reg<nt, vt>(shared, reg, tid, sync);
	}

	template <int nt, int vt, typename type, std::size_t size1, std::size_t size2>
	__device__ void
	reg_to_reg(const type reg1, type (&shared)[size1], type (&reg2)[size2],
			int tid, bool sync = true)
	{
		shared[tid] = reg1;
		__syncthreads();
		shared_to_reg<nt, vt>(shared, reg2, tid, sync);
	}
}
