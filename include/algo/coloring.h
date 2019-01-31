#pragma once
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "lwps/matrix.h"
#include "lwps/vector.h"
#include "util/launch.h"

namespace algo {
class coloring {
private:
	int sz;
	int ncolors;
	util::memory<int> cstarts;
protected:
	template <typename p_iterator>
	lwps::vector
	permute_with(const lwps::vector& v, p_iterator pdata) const
	{
		auto size = sz;
		if (size == 0) return v;
		(void) (lwps::size(v) + lwps::matrix_size{size, 1});
		lwps::vector permuted(size);
		auto* fdata = v.values();
		auto* gdata = permuted.values();

		auto k = [=] __device__ (int tid) { gdata[pdata[tid]] = fdata[tid]; };
		util::transform<128, 7>(k, sz);
		return std::move(permuted);
	}

	template <typename p_iterator, typename q_iterator>
	lwps::matrix
	permute_with(const lwps::matrix& m, p_iterator pdata, q_iterator qdata) const
	{
		auto size = sz;
		if (size == 0) return m;
		(void) (lwps::size(m) + lwps::matrix_size{size, size});
		auto nonzero = m.nonzero();

		lwps::matrix result{size, size, nonzero};
		auto* sdata = m.starts();
		auto* vdata = m.values();
		auto* idata = m.indices();
		auto* tdata = result.starts();
		auto* wdata = result.values();
		auto* jdata = result.indices();

		auto r = [=] __device__ (int tid)
		{
			auto p = pdata[tid];
			tdata[tid] = sdata[p+1] - sdata[p];
			if (!tid) tdata[size] = 0;
		};
		util::transform<128, 7>(r, size);

		thrust::execution_policy<thrust::system::cuda::tag> exec;
		thrust::exclusive_scan(exec, tdata, tdata + size + 1, tdata);

		auto v = [=] __device__ (int tid)
		{
			auto entries = tdata[tid+1] - tdata[tid];
			auto old_start = sdata[pdata[tid]];
			auto new_start = tdata[tid];

			for (int i = 0; i < entries; ++i) {
				jdata[new_start + i] = qdata[idata[old_start + i]];
				wdata[new_start + i] = vdata[old_start + i];
			}
			thrust::sort_by_key(thrust::seq, jdata + new_start,
					jdata + new_start + entries, wdata + new_start);
		};
		util::transform(v, size);

		return std::move(result);
	}
public:
	virtual lwps::matrix permute(const lwps::matrix&) const = 0;
	virtual lwps::vector permute(const lwps::vector&) const = 0;
	virtual lwps::matrix unpermute(const lwps::matrix&) const = 0;
	virtual lwps::vector unpermute(const lwps::vector&) const = 0;

	int size() const { return sz; }
	int colors() const { return ncolors; }
	int* starts() const { return cstarts.data(); }

	coloring(int size, int colors) :
		sz(size), ncolors(colors), cstarts(colors+1) {}
	virtual ~coloring() {}
};
}
