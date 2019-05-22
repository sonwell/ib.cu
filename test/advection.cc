#include <cmath>
#include "ins/advection.h"
#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"
#include "fd/grid.h"
#include "lwps/vector.h"

int
main(void)
{
	static constexpr auto pi = M_PI;
	static constexpr auto n = 4;
	constexpr fd::dimension x{1, fd::boundary::periodic()};
	constexpr fd::dimension y{1, fd::boundary::periodic()};
	constexpr fd::domain domain{fd::grid::mac(n), x, y};

	lwps::vector u(n * n);
	lwps::vector v(n * n);
	auto* uvalues = u.values();
	auto* vvalues = v.values();
	auto k = [=] __device__ (int tid)
	{
		auto i = tid % n;
		auto j = (tid / n) % n;
		uvalues[tid] = cos(2 * pi * i / n) * cos(2 * pi * (j + 0.5) / n);
		vvalues[tid] = cos(2 * pi * (i + 0.5) / n) * cos(2 * pi * j / n);
	};
	util::transform(k, n * n);
	std::cout << u << std::endl << v << std::endl;

	ins::advection advection{domain};
	auto&& h = advection(u, v);
	std::cout << std::get<0>(h) << std::endl << std::get<1>(h) << std::endl;

	return 0;
}
