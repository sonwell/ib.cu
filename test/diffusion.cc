#include <utility>
#include <cmath>

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"
#include "fd/grid.h"
#include "fd/size.h"

#include "ins/diffusion.h"

#include "algo/types.h"
#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "algo/preconditioner.h"

#include "util/functional.h"
#include "util/debug.h"

#include "cuda/event.h"
#include "units.h"

template <typename grid_type, typename fn_type>
decltype(auto)
fill(const grid_type& grid, fn_type fn)
{
	static constexpr auto dimensions = grid_type::dimensions;
	using algo::vector;

	auto x = [=] __device__ (int tid, auto h)
	{
		using namespace util::functional;
		auto i = tid;
		std::array<double, dimensions> x = {0.};
		auto p = [&] (double& x, const auto& comp)
		{
			auto l = comp.points();
			auto j = i % l;
			i /= l;
			x = (j + comp.shift()) / comp.resolution();
		};
		map(p, x, grid.components());
		return h(x);
	};

	vector v{fd::size(grid)};
	auto* data = v.values();
	auto f = [=] __device__ (int tid, auto g, auto h) { data[tid] = g(tid, h); };

	util::transform<128, 7>(f, fd::size(grid), x, fn);
	return v;
}

int
main(void)
{
	constexpr fd::dimension x{16_um, fd::boundary::dirichlet()};
	constexpr fd::domain domain{x};
	constexpr fd::centered mac{256};

	auto density = 1_g / (1_cm * 1_cm * 1_cm);
	auto viscosity = 1_cP;
	auto refinement = mac.refinement();
	auto scale = domain.unit();
	units::length h = scale / refinement;
	units::time k = 0.0000001_s;
	ins::simulation params{k, 1_s, scale, viscosity / density, 1e-11};

	util::set_default_resource(cuda::default_device().memory());

	util::logging::info("k: ", k);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", viscosity);
	util::logging::info("ρ: ", density);
	util::logging::info("λ: ", (k / (h * h)) * viscosity / density);

	constexpr fd::grid grid{mac, domain, x};
	ins::diffusion step{grid, params};
	auto quadratic = [=] __device__ (auto x) { return 0.5 * x[0] * (scale - x[0]); };
	auto constant = [=] __device__ (auto x) { return (double) (viscosity / density); };
	auto u = fill(grid, quadratic);
	auto b = fill(grid, constant);
	auto ub = 0 * u;

	std::cout << "import numpy as np\n";
	std::cout << "import matplotlib.pyplot as plt\n";
	std::cout << "n = " << mac.refinement() << '\n';
	std::cout << "x = (np.arange(n) + 0.5) / n\n";
	std::cout << "u0 = " << linalg::io::numpy << u << '\n';
		//<< "plt.plot(x, u0)\n";
	for (int i = 0; i < 1000; ++i) {
		u = step(1.0, u, ub, b);
		if (i % 100 == 0)
		std::cout << "u" << (i+1) << " = " << linalg::io::numpy << u << '\n'
			<< "plt.plot(x, u" << (i+1) << ")\n";
	}

	std::cout << "plt.show()\n";
	return 0;
}
