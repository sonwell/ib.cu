#include <fstream>
#include <cstdio>
#include <unistd.h>
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
#include "bases/gmq.h"
#include "bases/phs.h"
#include "bases/scaled.h"
#include "bases/polynomials.h"
#include "bases/geometry.h"
#include "ib/novel.h"
#include "ib/cosine.h"
#include "forces/skalak.h"
#include "forces/bending.h"
#include "forces/repelling.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "cuda/event.h"
#include "units.h"
#include "bubble.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	static constexpr int steps_per_print = 100;
	std::ostream& output = std::cout;
	int count = 0;

	template <typename u_type, typename ub_type>
	void
	operator()(const u_type& u, const ub_type& ub, const vector& p, const matrix& x)
	{
		if ((count++) % steps_per_print) return;
		using namespace util::functional;
		auto store = [&] (auto&& v) { output << linalg::io::binary << v; };
		map(store, u);
		map(store, ub);
		store(p);
		store(x);
	}

	binary_writer(std::ostream& output = std::cout) :
		output(output) {}
};

struct null_writer {
	null_writer(std::ostream& = std::cout) {}

	template <typename u_type, typename ub_type>
	void operator()(const u_type&, const ub_type&, const vector&, const matrix&) {}
};

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

template <typename grid_type, typename fn_type>
decltype(auto)
fill_flow(const grid_type& grid, fn_type fn)
{
	auto x = [=] __device__ (int tid, auto h)
	{
		using namespace util::functional;
		auto i = tid;
		std::array<double, grid_type::dimensions> x = {0.};
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

template <typename grid_type, typename domain_type, typename reference_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain,
			const reference_type& ref, units::unit<0, 0, -1> shear_rate)
{
	constexpr auto stretch = bases::shear({{1, 0, 0}, {0, 1/1.1, 0}, {0, 0, 1.1}});
	constexpr auto translate = bases::translate({  8_um,  8_um,  8_um});
	bases::container cells{ref, stretch | translate};
	matrix x = cells.x;

	auto velocity = zeros(grid, domain);
	auto boundary_velocity = zeros(grid, domain);
	auto r = grid.refinement();

	vector p{r * r * r, linalg::zero};

	return std::make_tuple(
		std::move(velocity),
		std::move(boundary_velocity),
		std::move(p),
		std::move(x)
	);
}

int
main(int argc, char** argv)
{
	int iterations = (argc > 1) ? atoi(argv[1]) : 10;
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic};
	constexpr fd::dimension y{16_um, fd::boundary::periodic};
	constexpr fd::dimension z{16_um, fd::boundary::periodic};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{GRID_REFINEMENT};

	constexpr auto shear_rate = 1000 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto k = 0.0000016_s * (h / 1_um) * (h / 1_um);
	constexpr auto viscosity = 1_cP;
	constexpr auto density = 1_g / 1_mL;
	constexpr ins::parameters params {k, time_scale, length_scale, density, viscosity, 1e-11};

	constexpr forces::skalak forces{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("k: ", params.timestep);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);
	util::logging::info("λ: ", params.coefficient * k / (h * h));

	constexpr ib::delta::cosine phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> basic;
	bubble ref{NUM_DATA_SITES, NUM_SAMPLE_SITES, basic};

	auto [u, ub, p, rx] = initialize(mac, domain, ref, shear_rate);
	bases::container cells{ref, std::move(rx)};

	auto [rows, cols] = linalg::size(cells.x);
	auto n = rows * cols / domain.dimensions;
	ins::solver step{mac, domain, params};
	units::time t = 0;

	binary_writer write;
	auto f = [&] (units::time t0, const auto& v)
	{
		auto k = t0 - t;
		auto& x = cells.geometry(bases::current).data.position;
		auto n = x.rows() * x.cols() / domain.dimensions;
		auto w = interpolate(n, x, v);
		auto z = (double) k * std::move(w) + x;

		bases::container tmp{ref, std::move(z)};
		auto& y = tmp.geometry(bases::current).sample.position;
		auto m = y.rows() * y.cols() / domain.dimensions;
		auto f = forces(tmp);
		auto g = spread(m, y, f);
		return g;
	};

	for (int i = 0; i < iterations; ++i) {
		util::logging::info("simulation time: ", t);
		try {
			auto [tn, un, pn] = step(t, std::move(u), ub, f);
			t = tn;
			u = std::move(un);
			p = std::move(pn);
		}
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return -1;
		}
		auto v = (double) k * interpolate(n, cells.x, u);
		cells.x += std::move(v);
		//write(u, ub, p, cells.x);
	}
	//write(u, ub, p, cells.x);
	util::logging::info("simulation time: ", t);


	return 0;
}
