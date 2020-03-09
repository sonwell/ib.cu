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
	constexpr fd::dimension x{16_um, fd::boundary::periodic()};
	constexpr fd::dimension y{16_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y};
	constexpr fd::mac cell{n};
	double unit = domain.unit();

	fd::vector u(n * n);
	fd::vector v(n * n);
	fd::vector ue(n * n);
	fd::vector ve(n * n);
	auto* uvalues = u.values();
	auto* vvalues = v.values();
	auto* euvalues = ue.values();
	auto* evvalues = ve.values();
	auto k = [=] __device__ (int tid)
	{
		auto i = tid % n;
		auto j = (tid / n) % n;
		uvalues[tid] = sin(2 * pi * i / n) * sin(2 * pi * (j + 0.5) / n);
		vvalues[tid] = cos(2 * pi * (i + 0.5) / n) * cos(2 * pi * j / n);
		euvalues[tid] = 2 * pi / unit * sin(2 * pi * i / n) * cos(2 * pi * i / n);
		evvalues[tid] = -2 * pi / unit * sin(2 * pi * j / n) * cos(2 * pi * j / n);
	};
	util::transform(k, n * n);

	std::cout << "import numpy as np\n";
	std::cout << "u = " << linalg::io::numpy << ue << '\n';
	std::cout << "v = " << linalg::io::numpy << ve << '\n';

	ins::advection advection{fd::mac(n), domain};
	auto [hx, hy] = advection(u, v);
	std::cout << "hu = " << linalg::io::numpy << u << '\n';
	std::cout << "hv = " << linalg::io::numpy << v << '\n';

	fd::grid gx{cell, domain, x};
	fd::grid gy{cell, domain, y};

	auto axx = fd::average(gx, x);
	auto axy = fd::average(gx, y);
	auto ayx = fd::average(gy, x);
	auto ayy = fd::average(gy, y);

	auto uax = axx * u;
	auto uay = axy * u;
	auto vax = ayx * v;
	auto vay = ayy * v;

	auto uu = uax % uax;
	auto uv = uay % vax;
	auto vv = vay % vay;

	fd::grid gxx{fd::shift::directionally<0>(cell), domain, x};
	fd::grid gyy{fd::shift::directionally<1>(cell), domain, y};

	auto dxx = fd::differential(gxx, x);
	auto dxy = fd::differential(gxx, y);
	auto dyx = fd::differential(gyy, x);
	auto dyy = fd::differential(gyy, y);

	std::cerr << linalg::io::numpy << ayy << '\n';

	std::cout << "hue = " << linalg::io::numpy << dxx * uu + dxy * uv << '\n';
	std::cout << "hve = " << linalg::io::numpy << dyx * uv + dyy * vv << '\n';

	/*std::cout << "print(hue - hu)\n";
	std::cout << "print(hve - hv)\n";*/

	std::cout << "from mayavi import mlab\n";
	std::cout << "n = " << n << '\n';
	std::cout << "r = " << (double) domain.unit() / cell.refinement() << '\n';
	std::cout << "x, y = np.mgrid[0:n, 0:n]\n";

	std::cout << "mlab.surf(r * x, r * y, hue.reshape((n, n), order='F'))\n";
	std::cout << "mlab.show()\n";
	std::cout << "mlab.surf(r * x, r * y, hve.reshape((n, n), order='F'))\n";
	std::cout << "mlab.show()\n";

	return 0;
}
