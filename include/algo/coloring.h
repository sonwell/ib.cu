#pragma once
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "util/launch.h"
#include "types.h"

namespace algo {

class coloring {
private:
	int sz;
	int ncolors;
	int* cstarts;
	//util::memory<int> cstarts;
protected:
	template <typename p_iterator>
	vector
	permute_with(const vector& v, p_iterator pdata) const
	{
		using algo::size;
		auto n = sz;
		(void) (size(v) + size{n, 1});
		vector permuted(n);
		auto* fdata = v.values();
		auto* gdata = permuted.values();

		auto k = [=] __device__ (int tid) { gdata[pdata[tid]] = fdata[tid]; };
		util::transform<128, 7>(k, n);
		return permuted;
	}

	template <typename p_iterator, typename q_iterator>
	matrix
	permute_with(const matrix& m, p_iterator pdata, q_iterator qdata) const
	{
		using algo::size;
		auto n = sz;
		(void) (size(m) + size{n, n});
		auto nonzero = m.nonzero();

		matrix result{n, n, nonzero};
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
			if (!tid) tdata[n] = 0;
		};
		util::transform<128, 7>(r, n);

		thrust::execution_policy<thrust::system::cuda::tag> exec;
		thrust::exclusive_scan(exec, tdata, tdata + n + 1, tdata);

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
		util::transform(v, n);
		return result;
	}
public:
	virtual matrix permute(const matrix&) const = 0;
	virtual vector permute(const vector&) const = 0;
	virtual matrix unpermute(const matrix&) const = 0;
	virtual vector unpermute(const vector&) const = 0;

	int size() const { return sz; }
	int colors() const { return ncolors; }
	int* starts() const { return cstarts; }

	coloring(int sz, int colors) :
		sz(sz), ncolors(colors), cstarts(new int[colors+1]) {}
	virtual ~coloring() {}
};

} // namespace algo
