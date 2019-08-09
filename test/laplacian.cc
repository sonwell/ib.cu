#include <iostream>
#include "util/memory_resource.h"
#include "cuda/device.h"

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/average.h"
#include "fd/laplacian.h"
#include "ins/boundary.h"
#include "mg/interpolation.h"

int
main(void)
{
	constexpr fd::dimension x(100_mm, fd::boundary::periodic());
	constexpr fd::dimension y(100_mm, fd::boundary::dirichlet());
	constexpr fd::dimension z(100_mm, fd::boundary::periodic());

	constexpr fd::domain domain(fd::centered(8), x, y, z);

	auto&& dev = cuda::default_device();
	util::set_default_resource(dev.memory());

	auto&& lap = mg::interpolation(domain, x);
	std::cout << lap << std::endl;

	//auto k = [=] __device__ (int tid)
	//{
	//	auto res = domain.resolution();
	//	auto r = domain.clamp({200_mm, 100_mm, 200_mm});
	//	printf("%d %f %f %f\n", tid, r[0] * res, r[1] * res, r[2] * res);
	//};
	//util::transform(k, 1);
	//cuda::synchronize();
	return 0;
}
