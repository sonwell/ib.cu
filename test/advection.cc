#include <cmath>
#include "ins/advection.h"
#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"
#include "fd/grid.h"
#include "fd/types.h"
#include "util/memory_resource.h"
#include "cuda/device.h"

int
main(void)
{
	util::set_default_resource(cuda::default_device().memory());

	static constexpr auto pi = M_PI;
	static constexpr auto n = 8;
	constexpr fd::dimension x{1, fd::boundary::periodic()};
	constexpr fd::dimension y{1, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y};

	fd::vector u(n * n);
	fd::vector v(n * n);
	auto* uvalues = u.values();
	auto* vvalues = v.values();
	auto k = [=] __device__ (int tid)
	{
		auto i = tid % n;
		auto j = (tid / n) % n;
		uvalues[tid] = sin(2 * pi * i / n) * sin(2 * pi * (j + 0.5) / n);
		vvalues[tid] = cos(2 * pi * (i + 0.5) / n) * cos(2 * pi * j / n);
	};
	util::transform(k, n * n);
	std::cout << u << std::endl << v << std::endl;

	ins::advection advection{fd::mac(n), domain};
	auto [hx, hy] = advection(u, v);
	std::cout << hx << std::endl << hy << std::endl;

	return 0;
}
