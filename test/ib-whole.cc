#include <fstream>
#include <cstdio>
#include <algorithm>
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
#include "ib/bspline.h"
#include "ib/solver.h"
#include "ib/state.h"

#include "forces/skalak.h"
#include "forces/neohookean.h"
#include "forces/bending.h"
#include "forces/dissipation.h"
#include "forces/tether.h"
#include "forces/damping.h"
#include "forces/repelling.h"
#include "forces/combine.h"

#include "units.h"
#include "rbc.h"
#include "platelet.h"
#include "endothelium.h"

using bases::matrix;
using bases::vector;

namespace setup {

struct setup {
	units::unit<0, 0, -1> shear;
	endothelial_shape shape;
};

constexpr setup flat{1000/1_s, endothelial_shape::flat};
constexpr setup fast{1000/1_s, endothelial_shape::elongated};
constexpr setup slow{50/1_s, endothelial_shape::cobblestone};

} // namespace setup

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.1_ms;
	std::ostream& output = std::cout;

	template <std::size_t dimensions, std::size_t n>
	void
	operator()(const ib::state<dimensions, n>& state)
	{
		const auto& t = state.t;
		if (t < time) return;
		time += interval;

		output << linalg::io::binary << state;
	}
};

struct null_writer {
	null_writer(units::time = 0, units::time = 0,
			std::ostream& = std::cout) {}

	template <std::size_t dimensions, typename ... object_types>
	void operator()(const ins::state<dimensions>&,
			const object_types& ...) {}
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
	return apply([] (auto ... v) { return std::array{std::move(v)...}; },
	             map(k, fd::components(domain)));
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

template <typename grid_type, typename domain_type>
decltype(auto)
boundary_velocity(const grid_type& grid, const domain_type& domain,
		units::unit<0, 0, -1> shear_rate/*, units::time t*/)
{
	using fd::correction::second_order;

	//double milliseconds = t / 1_ms;
	//double angle = std::min(1.0, milliseconds / 7.2) * M_PI_4;
	constexpr double angle = M_PI_4;
	auto&& [x, y, z] = domain.components();
	auto ymax = y.length();
	double twv = ymax * shear_rate;
	fd::grid gx{grid, domain, x};
	fd::grid gz{grid, domain, z};
	auto ub = zeros(grid, domain);

	auto bx = fd::upper_boundary(gx, y, second_order);
	auto bz = fd::upper_boundary(gz, y, second_order);
	ub[0] = bx * vector{fd::size(gx, y), algo::fill(sin(angle) * twv)};
	ub[2] = bz * vector{fd::size(gz, y), algo::fill(cos(angle) * twv)};

	return ub;
}

template <typename grid_type, typename domain_type, bases::meta::traits traits>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain,
			const rbc& rbc, const platelet& plt, traits, const char* fname)
{
	using state = ib::state<domain_type::dimensions, 3>;
	matrix rbcs, plts, endo;

	if (fname == nullptr) {
		rbcs = bases::shape(rbc,
			bases::rotate(5.068183442568522, {1.030437666936895, -0.021108450766973392, -0.04630059618399827}) | bases::translate({4.015214448901787_um, 6.100138060189724_um, 1.9880461041978537_um}),
			bases::rotate(5.007815431576859, {0.9792381457522887, 0.06271914842308295, -0.05642529490935586}) | bases::translate({11.877848594136138_um, 5.83714041009_um, 3.97975684227_um}),
			bases::rotate(5.074088738134238, {1.0142231005007727, -2.9623760263254767e-05, -0.039362970458103955}) | bases::translate({4.146345896973395_um, 5.881458848974533_um, 5.962170272743408_um}),
			bases::rotate(5.079676113839819, {0.9890875402932139, -0.031109409809641483, -0.05032173371464803}) | bases::translate({12.013925005641164_um, 6.158047729278891_um, 7.93432970522188_um}),
			bases::rotate(5.1362464926490805, {1.0348848942058133, 0.01720012363571293, -0.11947361656641693}) | bases::translate({3.908805848485636_um, 6.091134893946915_um, 9.855992001042118_um}),
			bases::rotate(5.054989211831597, {0.9675754189340313, -0.048388248918717285, 0.03738865256187718}) | bases::translate({12.083267206719906_um, 5.80425714837_um, 12.1799588627_um}),
			bases::rotate(5.108732820408364, {0.9569821581817692, -0.056252540193673733, -0.07872719696376375}) | bases::translate({4.110024751201524_um, 5.91689772487407_um, 13.976372295071807_um}),
			bases::rotate(4.931079097988467, {0.9593196951752679, -0.020300578072308676, 0.07572396922527558}) | bases::translate({12.02165847715475_um, 6.012459328852655_um, 15.96571613373596_um})//,
			//bases::rotate(5.906662195399711, {0.9971582629235043, -0.021070905117683534, -0.08102102562217757}) | bases::translate({3.894077067182094_um, 12.726280756282677_um, 4.050277401672479_um}),
			//bases::rotate(5.9110196777911215, {0.947093283898209, 0.11996399058192268, -0.028579726936423924}) | bases::translate({11.978506744976466_um, 12.665694767718799_um, 7.972075324441059_um}),
			//bases::rotate(5.915284085003262, {0.8577444450409515, -0.10649820699097605, 0.4265386595355442}) | bases::translate({3.9603486846147624_um, 12.680732811531806_um, 12.240818745581036_um}),
			//bases::rotate(5.872101056788503, {0.9257797018884111, 0.19176928816818867, 0.11104080010931071}) | bases::translate({11.866323151661556_um, 12.605616419550312_um, 16.071587706914382_um})
		);
		plts = bases::shape(plt, bases::translate({0_um, 1.75_um, 0_um}), bases::translate({8_um, 2.75_um, 8_um}));
		endo = traits::shape(traits::sample(16000));
		fd::grid g{fd::shift::diagonally(grid), domain};
		return state{0_s, zeros(grid, domain), {fd::size(g), linalg::zero},
			         std::array{std::move(rbcs), std::move(plts), std::move(endo)}};
	}

	std::fstream f(fname, std::ios::in | std::ios::binary);
	state st;
	f >> linalg::io::binary >> st;
	return st;
}

int
main(int argc, char** argv)
{
	constexpr auto setup = setup::fast;

	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic};
	constexpr fd::dimension y{12_um, fd::boundary::dirichlet};
	constexpr fd::dimension z{16_um, fd::boundary::periodic};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{20};

	constexpr auto shear_rate = setup.shear;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto kmin = 50_ns;
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-11};

	constexpr forces::skalak rbc_tension{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
	constexpr forces::bending rbc_bending{2e-12_erg};
	constexpr forces::dissipation rbc_dissip{2.5e-7_dyn*1_s/1_cm};
	constexpr forces::combine rbc_forces{rbc_tension, rbc_bending, rbc_dissip};

	constexpr forces::neohookean plt_tension{1e-1_dyn/1_cm, 1_dyn/1_cm};
	constexpr forces::bending plt_bending{2e-11_erg, true};
	constexpr forces::combine plt_forces{plt_tension, plt_bending};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);

	constexpr ib::delta::bspline<3> phi;
	ib::solver step{mac, domain, phi, params};

	constexpr bases::phs<7> sharp;
	using endo = endothelium<setup.shape>;
	rbc rbc{2500, 10000, sharp};
	platelet plt{900, 900, sharp};


	auto st = initialize(mac, domain, rbc, plt,
			bases::traits<endo>{}, argc > 1 ? argv[1] : nullptr);
	auto ub = boundary_velocity(mac, domain, shear_rate);
	auto ex0 = endo::shape(endo::sample(st.x[2].rows()));

	units::time tmax = 100_ms;
	binary_writer write{st.t};

	auto spring = [&] (const matrix& ex, const matrix& eu)
	{
		constexpr auto spring = 2.5e+0_dyn/1_cm;
		constexpr auto damping = 1e-7_dyn*1_s/1_cm;
		auto f = -((double) spring * (ex - ex0) + (double) damping * eu);
		return std::make_pair(ex, f);
	};

	auto forces = [&] (units::time t, const auto& x, const auto& u)
	{
		const auto& [rx, px, ex] = x;
		const auto& [ru, pu, eu] = u;

		const bases::container rbcs{rbc, rx};
		const bases::container plts{plt, px};

		auto& ry = rbcs.geometry(bases::current).sample.position;
		auto rf = rbc_forces(rbcs, ru);

		auto& py = plts.geometry(bases::current).sample.position;
		auto pf = plt_forces(plts, pu);

		auto&& [ey, ef] = spring(ex, eu);

		return std::pair{
			std::array{ry, py, ey},
			std::array{rf, pf, ef}
		};
	};

	write(st);
	while (st.t < tmax) {
		util::logging::info("simulation time: ", st.t);
		try { st = step(std::move(st), ub, forces); }
		catch (std::system_error& e) {
			util::logging::error(e.what());
			return -e.code().value();
		}
		catch(std::runtime_error& e) {
			util::logging::error(e.what());
			return 255;
		}
		write(st);
	}
	return 0;
}
