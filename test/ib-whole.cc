#include <fstream>
#include <cstdio>
#include <unistd.h>
#include "util/memory_resource.h"
#include "cuda/device.h"
#include "cuda/timer.h"
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
#include "ib/roma.h"
#include "ib/bspline.h"
#include "forces/skalak.h"
#include "forces/neohookean.h"
#include "forces/bending.h"
#include "forces/tether.h"
#include "forces/repelling.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "cuda/event.h"
#include "units.h"
#include "platelet.h"
#include "rbc.h"
#include "endothelium.h"

using bases::matrix;
using bases::vector;

struct state {
	vector u, v, w;
	vector ub, vb, wb;
	vector p;
	matrix r, e; // rbcs, endothelium
};


std::ostream&
operator<<(std::ostream& out, const state& st)
{
	auto store = [&] (auto&& v) { out << linalg::io::binary << v; };
	store(st.u);
	store(st.v);
	store(st.w);
	store(st.ub);
	store(st.vb);
	store(st.wb);
	store(st.p);
	store(st.r);
	store(st.e);

	return out;
}

std::istream&
operator>>(std::istream& in, state& st)
{
	auto load = [&] (auto&& v) { in >> linalg::io::binary >> v; };
	load(st.u);
	load(st.v);
	load(st.w);
	load(st.ub);
	load(st.vb);
	load(st.wb);
	load(st.p);
	load(st.r);
	load(st.e);
	return in;
}

struct binary_writer {
	static constexpr int steps_per_print = 1000;
	std::ostream& output = std::cout;
	int count = 0;

	template <typename u_type, typename ub_type>
	void
	operator()(const u_type& u, const ub_type& ub, const vector& p,
			const matrix& x, const matrix& y, bool force = false)
	{
		if ((count++) % steps_per_print && !force) return;
		using namespace util::functional;
		auto store = [&] (auto&& v) { output << linalg::io::binary << v; };
		map(store, u);
		map(store, ub);
		store(p);
		store(x);
		store(y);
	}

	void operator()(const state& st) { output << st; }

	binary_writer(std::ostream& output = std::cout) :
		output(output) {}
};

struct null_writer {
	null_writer(std::ostream& = std::cout) {}

	template <typename u_type, typename ub_type>
	void operator()(const u_type&, const ub_type&, const vector&, const matrix&, const matrix&) {}
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

template <typename grid_type, typename domain_type, typename ref_type>
state
initialize(const grid_type& grid, const domain_type& domain,
			units::unit<0, 0, -1> shear_rate, const ref_type& ref,
			const char* fname)
{
	state st;

	if (fname == nullptr) {
		auto&& [x, y, z] = domain.components();
		fd::grid g{grid, domain, z};
		auto r = grid.refinement();

		auto ymax = y.length();
		double twv = ymax * shear_rate;
		auto shear = [=] __device__ (const auto& x) { return twv * x[1] / ymax; };
		auto b = fd::upper_boundary(g, y, fd::correction::second_order);

		std::tie(st.u, st.v, st.w) = zeros(grid, domain);
		std::tie(st.ub, st.vb, st.wb) = zeros(grid, domain);
		st.w = fill_flow(g, shear);
		st.wb = b * vector{fd::size(g, y), algo::fill(twv)};
		st.p = vector{r * r * r, linalg::zero};

		bases::container cells{ref,
			bases::rotate(5.068183442568522, {1.030437666936895, -0.021108450766973392, -0.04630059618399827}) | bases::translate({4.015214448901787_um, 6.100138060189724_um, 1.9880461041978537_um}),
			bases::rotate(5.007815431576859, {0.9792381457522887, 0.06271914842308295, -0.05642529490935586}) | bases::translate({11.877848594136138_um, 5.83714041009_um, 3.97975684227_um}),
			bases::rotate(5.074088738134238, {1.0142231005007727, -2.9623760263254767e-05, -0.039362970458103955}) | bases::translate({4.146345896973395_um, 5.881458848974533_um, 5.962170272743408_um}),
			bases::rotate(5.079676113839819, {0.9890875402932139, -0.031109409809641483, -0.05032173371464803}) | bases::translate({12.013925005641164_um, 6.158047729278891_um, 7.93432970522188_um}),
			bases::rotate(5.1362464926490805, {1.0348848942058133, 0.01720012363571293, -0.11947361656641693}) | bases::translate({3.908805848485636_um, 6.091134893946915_um, 9.855992001042118_um}),
			bases::rotate(5.054989211831597, {0.9675754189340313, -0.048388248918717285, 0.03738865256187718}) | bases::translate({12.083267206719906_um, 5.80425714837_um, 12.1799588627_um}),
			bases::rotate(5.108732820408364, {0.9569821581817692, -0.056252540193673733, -0.07872719696376375}) | bases::translate({4.110024751201524_um, 5.91689772487407_um, 13.976372295071807_um}),
			bases::rotate(4.931079097988467, {0.9593196951752679, -0.020300578072308676, 0.07572396922527558}) | bases::translate({12.02165847715475_um, 6.012459328852655_um, 15.96571613373596_um}),
			//bases::rotate(5.906662195399711, {0.9971582629235043, -0.021070905117683534, -0.08102102562217757}) | bases::translate({3.894077067182094_um, 12.726280756282677_um, 4.050277401672479_um}),
			//bases::rotate(5.9110196777911215, {0.947093283898209, 0.11996399058192268, -0.028579726936423924}) | bases::translate({11.978506744976466_um, 12.665694767718799_um, 7.972075324441059_um}),
			//bases::rotate(5.915284085003262, {0.8577444450409515, -0.10649820699097605, 0.4265386595355442}) | bases::translate({3.9603486846147624_um, 12.680732811531806_um, 12.240818745581036_um}),
			//bases::rotate(5.872101056788503, {0.9257797018884111, 0.19176928816818867, 0.11104080010931071}) | bases::translate({11.866323151661556_um, 12.605616419550312_um, 16.071587706914382_um})
		};
		//bases::container ecs{endothelium, id};
		matrix tmp = cells.x;
		st.r = std::move(tmp);
		return st;
	}

	std::fstream f(fname, std::ios::in | std::ios::binary);
	f >> st;
	return st;
}

int
main(int argc, char** argv)
{
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic};
	constexpr fd::dimension y{16_um, fd::boundary::dirichlet};
	constexpr fd::dimension z{16_um, fd::boundary::periodic};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{128};

	constexpr auto shear_rate = 1000 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto kmin = 0.0000020_s * (h / 1_um) * (h / 1_um);
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-11};

	constexpr forces::skalak rbc_tension{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
	constexpr forces::bending rbc_bending{2e-12_erg};
	constexpr forces::combine rbc_forces{rbc_tension, rbc_bending};

	/*constexpr forces::tether ec_tether{2.45e-4_dyn/1_cm};
	constexpr forces::neohookean ec_tension{1e-7_dyn/1_cm, 0};
	constexpr forces::combine ec_forces{ec_tether, ec_tension};*/

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);

	constexpr ib::delta::bspline<2> phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> sharp;
	rbc rbc{1600, 10000, sharp};
	//endothelium endothelium{4096, 4900, sharp};

	auto st = initialize(mac, domain, shear_rate, rbc, argc > 1 ? argv[1] : nullptr);
	std::tuple u = {std::move(st.u), std::move(st.v), std::move(st.w)};
	std::tuple ub  = {std::move(st.ub), std::move(st.vb), std::move(st.wb)};
	vector p = std::move(st.p);
	bases::container rbcs{rbc, std::move(st.r)};
	/*bases::container ecs{endothelium, std::move(st.e)};*/

	ins::solver step{mac, domain, params};

	binary_writer write;
	units::time t = 0, tmax = 1_ms, k = kmin;

	auto forces_of = [&] (const auto& cell, const auto& forces,
			const units::time k, const auto& u)
	{
		using bases::current;
		auto& x = cell.geometry(current).data.position;
		auto [n, m] = linalg::size(x);
		auto w = interpolate(n * m / domain.dimensions, x, u);
		auto z = (double) k * std::move(w) + x;

		const auto& ref = bases::ref(cell);
		bases::container tmp{ref, std::move(z)};
		auto f = forces(cell);
		auto& y = tmp.geometry(current).sample.position;
		auto [r, s] = linalg::size(y);
		return spread(r * s / domain.dimensions, y, f);
	};

	auto f = [&] (units::time tn, const auto& v)
	{
		using namespace util::functional;
		units::time dt = tn - t;
		auto add = partial(map, std::plus<vector>{});
		auto k = [&] (const auto& pair)
		{
			auto& [obj, fn] = pair;
			return forces_of(obj, fn, dt, v);
		};
		auto op = [&] (const auto& l, const auto& r) { return add(l, k(r)); };
		auto m = [&] (const auto& f, const auto& ... r) { return foldl(op, k(f), r...); };
		return apply(m, zip(std::forward_as_tuple(rbcs/*, ecs*/),
		                    std::forward_as_tuple(rbc_forces/*, ec_forces*/)));
	};

	int err = 0;
	write(u, ub, p, rbcs.x, matrix{} /*ecs.x*/);
	while (t < tmax) {
		util::logging::info("simulation time: ", t);
		try {
			auto [tn, un, pn] = step(t, std::move(u), ub, f);
			k = tn - t;
			t = tn;
			u = std::move(un);
			p = std::move(pn);
		}
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			err = 255;
			break;
		}
		auto m = [&, &u=u] (auto& cell)
		{
			auto [n, m] = linalg::size(cell.x);
			auto v = interpolate(n * m / domain.dimensions, cell.x, u);
			cell.x += (double) k * std::move(v);
		};
		m(rbcs); /*m(ecs);*/
		write(u, ub, p, rbcs.x, matrix{} /*ecs.x*/);
	}
	write(u, ub, p, rbcs.x, matrix{} /*ecs.x*/, true);

	return err;
}
