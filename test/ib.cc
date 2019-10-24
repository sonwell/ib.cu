#include "util/memory_resource.h"
#include "cuda/device.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/domain.h"
#include "fd/boundary_ops.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "bases/types.h"
#include "bases/container.h"
#include "bases/transforms.h"
#include "bases/phs.h"
#include "bases/polynomials.h"
#include "bases/geometry.h"
#include "ib/sweep.h"
#include "ib/spread.h"
#include "ib/interpolate.h"
#include "forces/bending.h"
#include "forces/skalak.h"
#include "forces/neohookean.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

template <typename tag_type, typename domain_type>
decltype(auto)
zeros(const tag_type& tag, const domain_type& domain)
{
	using namespace util::functional;
	auto k = [&] (const auto& comp) {
		fd::grid g{tag, domain, comp};
		return algo::vector{fd::size(g), linalg::zero};
	};
	return map(k, fd::components(domain));
}

template <typename grid_type>
decltype(auto)
shear_flow(const grid_type& grid, double lower, double upper)
{
	auto k = [=] __device__ (int tid)
	{
		return lower + ((tid / 32) % 32 + 0.5) * (upper - lower) / 32;
	};
	return vector{fd::size(grid), algo::fill(k)};
}

int
main(int argc, char** argv)
{
	int iterations;
	if (argc == 1)
		iterations = 10;
	else
		iterations = atoi(argv[1]);
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{10_um, fd::boundary::periodic()};
	constexpr fd::dimension y{10_um, fd::boundary::dirichlet()};
	constexpr fd::dimension z{10_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{64};

	constexpr auto twv = 1_cm / 1_s;
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr ins::parameters params {/*4.096 * (double) (h * h)*/ 0.4 * (double) h, 1_g / 1_mL, 1_cP, 1e-8};

	util::logging::info("characteristic length: ", domain.unit());
	util::logging::info("h: ", h);
	util::logging::info("timestep: ", params.timestep);
	util::logging::info("viscosity: ", params.viscosity);
	util::logging::info("density: ", params.density);
	util::logging::info("diffusivity: ", params.viscosity / params.density);
	util::logging::info("lambda: ", params.timestep * params.viscosity / (params.density * domain.unit() * domain.unit()));
	util::logging::info("top wall velocity: ", twv, "k̂");

	constexpr ib::spread spread{mac, domain};
	constexpr ib::interpolate interpolate{mac, domain};

	constexpr bases::polyharmonic_spline<7> basic;
	rbc ref{625, 10000, basic};
	bases::container rbcs{ref,
		bases::shift({5_um, 5_um, 5_um}),
//		bases::shift({10_um, 10_um, 15_um})
	};
	matrix& cx = rbcs.x;
	auto n = cx.rows() * cx.cols() / domain.dimensions;
	double k = params.timestep;

	matrix f_l;
	constexpr forces::skalak skalak{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
//	constexpr forces::bending bending{2e-12_erg};
	constexpr forces::combine forces{skalak/*, bending*/};
	util::logging::info("skalak info: shear = ", (double) (2.5e-3_dyn/1_cm), " bulk = ", (double) (2.5e-1_dyn/1_cm));
	util::logging::info("scaled radius: ", (double) 3.91_um);
	auto f = [&] (const auto& v)
	{
		auto& x = rbcs.geometry(bases::current).data.position;
		auto n = x.rows() * x.cols() / domain.dimensions;
		auto u = interpolate(n, x, v);

		bases::container tmp{ref, x + k * std::move(u)};
		auto& y = tmp.geometry(bases::current).sample.position;
		auto m = y.rows() * y.cols() / domain.dimensions;
		auto f = forces(tmp);
		auto g = spread(m, y, f);
		f_l = std::move(f);
		return g;
	};

	ins::solver step{mac, domain, params};

	auto u = zeros(mac, domain);
	auto ub = zeros(mac, domain);

	using fd::correction::second_order;
	constexpr fd::grid g{mac, domain, z};
	auto b = fd::upper_boundary(g, y, second_order);
	std::get<2>(ub) = b * vector{fd::size(g, y), algo::fill((double) twv)};
	std::get<2>(u) = shear_flow(g, 0, twv);

	std::cout << "import numpy as np\n";
	std::cout << "from mayavi import mlab\n";
	std::cout << "import rbc_plotter\n\n";
	std::cout << std::setprecision(15) << std::fixed;

	{
		std::cout << "x0 = " << linalg::io::numpy << rbcs.x << '\n';
		std::cout << "t0 = " << linalg::io::numpy << rbcs.geometry(bases::current).data.sigma << '\n';
		std::cout << "n0 = " << linalg::io::numpy << rbcs.geometry(bases::current).data.normal << '\n';
		std::cout << "y0 = " << linalg::io::numpy << rbcs.geometry(bases::current).sample.position << '\n';
		std::cout << "s0 = " << linalg::io::numpy << rbcs.geometry(bases::current).sample.sigma << '\n';
		std::cout << "f0 = " << linalg::io::numpy << forces(rbcs) << '\n';
		std::cout << "p = " << linalg::io::numpy << rbc::sample(625) << '\n';
		std::cout << "plotter = rbc_plotter.Plotter(rbc_plotter.Sphere, p, x0, y0, f0)\n";
	}

	static constexpr auto steps_per_print = 100;
	for (int i = 0; i < iterations; ++i) {
		util::logging::info("simulation time: ", i * params.timestep);
		u = step(u, ub, f);
		auto v = k * interpolate(n, rbcs.x, u);
		rbcs.x += v;

		if (!((i + 1) % steps_per_print)) {
			std::cout << "x = " << linalg::io::numpy << rbcs.x << '\n';
			std::cout << "y = " << linalg::io::numpy << rbcs.geometry(bases::current).sample.position << '\n';
			std::cout << "f = " << linalg::io::numpy << f_l << '\n';
			std::cout << "plotter.plot(x, y, f)\n";
		}
	}

	return 0;
}
