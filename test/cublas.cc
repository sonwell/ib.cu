#include <stdexcept>
#include <iostream>
#include "util/launch.h"
#include "util/memory.h"
#include "cuda/device.h"
#include "cuda/copy.h"
#include "cublas/geam.h"
#include "cublas/gemm.h"
#include "util/debug.h"
#include "linalg/linalg.h"

using matrix = linalg::matrix<linalg::dense<double>>;

template <typename fill_func_type>
void
fill(matrix& m, fill_func_type fn)
{
	auto* values = m.values();
	auto rows = m.rows();
	auto cols = m.cols();

	auto k = [=] __device__ (int tid, auto f)
	{
		auto i = tid % rows;
		auto j = tid / rows;
		values[tid] = f(i, j);
	};
	util::transform<128, 8>(k, rows * cols, fn);
}

int
main(void)
{
	auto& dev = cuda::default_device();
	util::set_default_resource(dev.memory());
	matrix a{10, 10};
	matrix b{10, 10};

	fill(a, [] __device__ (int, int) { return 1.0; });
	fill(b, [] __device__ (int, int) { return 2.0; });
	auto&& c = a - b;
	auto&& d = a * b;
	std::cout << c << std::endl;
	std::cout << d << std::endl;
}
