#pragma once

namespace cuda {

template <int nt, int vt, typename value_type, std::size_t sz>
__device__ void
mem_to_shared(const value_type* mem, value_type (&shared)[sz],
		int cta, int tid, int count, bool sync=true)
{
	enum { nv = nt * vt };
	static_assert(nv <= sz);
	for (int i = 0; i < vt; ++i) {
		auto j = nv * cta + nt * i + tid;
		if (tid < nt && j < count)
			shared[nt * i + tid] = mem[j];
	}
	if (sync)
		__syncthreads();
}

template <int nt, int vt, typename value_type, std::size_t sz>
__device__ void
shared_to_mem(const value_type (&shared)[sz], value_type* mem,
		int cta, int tid, int count, bool sync=true)
{
	enum { nv = nt * vt };
	static_assert(nv <= sz);
	for (int i = 0; i < vt; ++i) {
		auto j = nv * cta + nt * i + tid;
		if (tid < nt && j < count)
			mem[j] = shared[nt * i + tid];
	}
}

template <int nt, int vt, typename value_type, std::size_t ssz, std::size_t rsz>
__device__ void
shared_to_reg(const value_type (&shared)[ssz], value_type (&reg)[rsz],
		int tid, bool sync=true)
{
	static_assert(ssz >= nt * vt);
	static_assert(rsz >= vt);
	for (int i = 0; i < vt; ++i)
		reg[i] = shared[vt * tid + i];
	if (sync)
		__syncthreads();
}

template <int nt, int vt, typename value_type, std::size_t ssz, std::size_t rsz>
__device__ void
reg_to_shared(const value_type (&reg)[rsz], value_type (&shared)[ssz],
		int tid, bool sync=true)
{
	static_assert(ssz >= nt * vt);
	static_assert(rsz >= vt);
	for (int i = 0; i < vt; ++i)
		shared[vt * tid + i] = reg[i];
	if (sync)
		__syncthreads();
}

template <int nt, int vt, typename value_type, std::size_t ssz, std::size_t rsz>
__device__ void
mem_to_reg(const value_type* mem, value_type (&shared)[ssz], value_type (&reg)[rsz],
		int cta, int tid, int count, bool sync=true)
{
	mem_to_shared<nt, vt>(mem, shared, cta, tid, count);
	shared_to_reg<nt, vt>(shared, reg, tid, sync);
}

template <int nt, int vt, typename value_type, std::size_t ssz, std::size_t rsz>
__device__ void
reg_to_mem(value_type (&reg)[rsz], value_type (&shared)[ssz], const value_type* mem,
		int cta, int tid, int count, bool sync=true)
{
	reg_to_shared<nt, vt>(reg, shared, tid);
	shared_to_mem<nt, vt>(shared, mem, cta, tid, count, sync);
}

}
