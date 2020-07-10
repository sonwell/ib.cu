#pragma once
#include <utility>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "util/launch.h"
#include "util/memory.h"
#include "cuda/copy.h" // XXX
#include "types.h"

namespace algo {

// Apply Gershgorin's circle theorem to approximate the range of eigenvalues for
// matrix `m`.
inline std::pair<double, double>
gershgorin(const matrix& m)
{
	auto rows = m.rows();
	auto* starts = m.starts();
	auto* indices = m.indices();
	auto* values = m.values();
	util::memory<double> upper(rows); // Circle upper bounds
	util::memory<double> lower(rows); // Circle lower bounds
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

} // namespace algo
