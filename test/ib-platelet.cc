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
#include "ib/pmqe.h"
#include "ib/novel.h"
#include "ib/roma.h"
#include "forces/hookean.h"
#include "forces/bending.h"
#include "forces/repelling.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "cuda/event.h"
#include "units.h"
#include "platelet.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	static constexpr int steps_per_print = 100;
	std::ostream& output = std::cout;
	int count = 0;

	template <typename u_type, typename ub_type>
	void
	operator()(const u_type& u, const ub_type& ub, const matrix& x)
	{
		if ((count++) % steps_per_print) return;
		using namespace util::functional;
		auto store = [&] (auto&& v) { output << linalg::io::binary << v; };
		map(store, u);
		map(store, ub);
		store(x);
	}

	binary_writer(std::ostream& output = std::cout) :
		output(output) {}
};

struct null_writer {
	null_writer(std::ostream& = std::cout) {}

	template <typename u_type, typename ub_type>
	void operator()(const u_type&, const ub_type&, const matrix&) {}
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


template <typename shape_fn>
static matrix
shapes(matrix x, shape_fn f)
{
	auto m = x.rows();
	auto* xdata = x.values();
	auto k = [=] __device__ (int tid)
	{
		constexpr bases::composition array{bases::impl::arrayifier{}};
		std::array<double, 3> x;
		for (int i = 0; i < 3; ++i)
			x[i] = xdata[m * i + tid];
		x = (f | array)(x);
		for (int i = 0; i < 3; ++i)
			xdata[m * i + tid] = x[i];
	};
	util::transform<128, 3>(k, m);
	return x;
}

template <typename grid_type, typename domain_type, typename reference_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain,
			const reference_type& ref, units::unit<0, 0, -1> shear_rate)
{
	constexpr auto transa = bases::translate({ 8_um, 8_um, 8_um});
	constexpr auto transb = bases::translate({16_um, 8_um, 8_um});
	bases::container cells{ref, transa, transb};
	matrix x = cells.x;

	auto velocity = zeros(grid, domain);
	auto boundary_velocity = zeros(grid, domain);

	{
		auto&& [x, y, z] = domain.components();
		auto ymax = y.length();
		double twv = ymax * shear_rate / 2;
		auto shear = [=] __device__ (const auto& x) { return -twv + 2 * twv * x[1] / ymax; };
		using fd::correction::second_order;
		fd::grid g{grid, domain, z};
		auto b = fd::upper_boundary(g, y, second_order)
		       - fd::lower_boundary(g, y, second_order);
		std::get<2>(boundary_velocity) = b * vector{fd::size(g, y), algo::fill(twv)};
		std::get<2>(velocity) = fill_flow(g, shear);
	}

	return std::make_tuple(
		std::move(velocity),
		std::move(boundary_velocity),
		std::move(x)
	);
}

template <typename domain_type>
decltype(auto)
resume(const domain_type& domain, const char* filename)
{
	using namespace util::functional;
	matrix x;
	std::ifstream f;
	auto use_stdin = strcmp(filename, "-") == 0;
	if (!use_stdin) f.open(filename, std::ios::in | std::ios::binary);
	std::istream& input = use_stdin ? std::cin : f;

	auto load = [&] (auto&&) { vector v; input >> linalg::io::binary >> v; return v; };
	auto u = map(load, domain.components());
	auto ub = map(load, domain.components());
	input >> linalg::io::binary >> x;
	if (input.eof())
		throw std::runtime_error("expected initialization state");

	return std::make_tuple(std::move(u), std::move(ub), std::move(x));
}

struct timer {
	std::string id;
	cuda::event start, stop;

	timer(std::string id) :
		id(id) { start.record(); }
	~timer() {
		stop.record();
		util::logging::info(id, ": ", stop-start, "ms");
	}
};

int
main(int argc, char** argv)
{
	int iterations = (argc > 1) ? atoi(argv[1]) : 10;
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic()};
	constexpr fd::dimension y{16_um, fd::boundary::dirichlet()};
	constexpr fd::dimension z{16_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{64};

	constexpr auto shear_rate = 50 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto k = 0.0000016_s * (h / 1_um) * (h / 1_um);
	constexpr ins::parameters params {k, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-11};

	constexpr forces::hookean tension{1e-1_dyn/1_cm};
	constexpr forces::bending bending{2e-11_erg};
	constexpr forces::repelling repelling{2.5e-3_dyn/1_cm};
	constexpr forces::combine forces{tension, /*bending, repelling*/};

	util::logging::info("meter: ", units::m);
	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("k: ", params.timestep);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);
	util::logging::info("λ: ", params.coefficient * k / (h * h));

	util::logging::info("tension info: shear =  ", tension.shear);
	util::logging::info("bending info: modulus = ", bending.modulus);
	util::logging::info("repelling info: strength = ", repelling.modulus);

	constexpr ib::delta::roma phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> basic;
	constexpr bases::generalized_multiquadric<7> smooth{0.0001};
	platelet ref{361, 1552, basic, smooth};

	auto [u, ub, rx] = (argc > 2) ?
		resume(domain, argv[2]) :
		initialize(mac, domain, ref, shear_rate);
	bases::container cells{ref, std::move(rx)};

	auto [rows, cols] = linalg::size(cells.x);
	auto n = rows * cols / domain.dimensions;
	ins::solver step{mac, domain, params};

	binary_writer write;
	auto f = [&] (const auto& v)
	{
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

	timer t{"runtime"};
	write(u, ub, cells.x);
	for (int i = 0; i < iterations; ++i) {
		util::logging::info("simulation time: ", i * params.timestep);
		try { u = step(std::move(u), ub, f); }
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return -1;
		}
		auto v = (double) k * interpolate(n, cells.x, u);
		cells.x += std::move(v);
		write(u, ub, cells.x);
	}

	return 0;
}
