#include <cmath>
#include "mg/chebyshev.h"
#include "fd/domain.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/laplacian.h"
#include "util/launch.h"
#include "cuda/device.h"
#include "util/memory_resource.h"
#include "units.h"

int
main(void)
{
	auto& dev = cuda::default_device();
	util::set_default_resource(dev.memory());

	static constexpr auto n = 512;
	static constexpr auto pi = M_PI;
	constexpr fd::dimension x{100_um, fd::boundary::periodic()};
	constexpr fd::domain domain{fd::grid::mac(n), x};

	auto lap = fd::laplacian(domain, x);
	mg::chebyshev smoother(domain, lap);

	for (int j = 0; j < n; j += 2) {
		std::cout << "sin(" << j << " pi x)" << std::endl;
		algo::vector y(n);
		auto* values = y.values();
		auto k = [=] __device__ (int tid) {
			values[tid] = sin(j * pi * tid / n);
		};
		util::transform(k, n);

		auto&& b = lap * y;

		auto z = 0 * b;
		for (int i = 0; i < 2; ++i) {
			std::cout << "  iteration " << i << ": " << abs(z - y) << std::endl;
			z += smoother(b - lap * z);
		}
		std::cout << "  iteration " << 2 << ": " << abs(z - y) << std::endl;
	}


	return 0;
}
