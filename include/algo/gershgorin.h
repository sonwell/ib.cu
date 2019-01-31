#pragma once
#include <utility>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "util/launch.h"
#include "cuda/copy.h"
#include "device_ptr.h"

namespace algo
{
	inline std::pair<double, double>
	gershgorin(const lwps::matrix& m)
	{
		auto rows = m.rows();
		auto* starts = m.starts();
		auto* indices = m.indices();
		auto* values = m.values();
		mem::device_ptr<double> upper(rows);
		mem::device_ptr<double> lower(rows);
		auto* udata = upper.data();
		auto* ldata = lower.data();

		auto k = [=] __device__ (int tid)
		{
			auto start = starts[tid];
			auto end = starts[tid+1];

			double diag = 0;
			double offdiag = 0;
			for (int i = start; i < end; ++i) {
				double value = values[i];
				if (indices[i] == tid) diag = value;
				else offdiag += abs(value);
			}
			udata[tid] = diag + offdiag;
			ldata[tid] = diag - offdiag;
		};
		util::transform(k, rows);

		double max_val, min_val;
		thrust::execution_policy<thrust::system::cuda::tag> exec;
		auto max = thrust::max_element(exec, udata, udata+rows);
		auto min = thrust::min_element(exec, ldata, ldata+rows);
		cuda::dtoh(&max_val, max, 1);
		cuda::dtoh(&min_val, min, 1);

		return {min_val, max_val};
	};
}
