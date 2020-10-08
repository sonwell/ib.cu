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
#include "ib/roma.h"
#include "forces/skalak.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "cuda/event.h"
#include "units.h"
#include "platelet.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.1_ms;
	std::ostream& output = std::cout;

	template <std::size_t dimensions, typename ... object_types>
	void
	operator()(const ins::state<dimensions>& state,
			const object_types& ... objects)
	{
		const auto& t = state.t;
		if (t < time) return;
		time += interval;

		output << linalg::io::binary << state;
		((output << linalg::io::binary) << ... << (matrix&) objects.x);
	}
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

template <typename grid_type, typename domain_type, typename reference_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain,
			const reference_type& ref, units::unit<0, 0, -1> shear_rate)
{
	constexpr auto dimensions = domain_type::dimensions;
	auto translate = bases::shear({{2.0, 0.0}, {0.0, 0.5}}) | bases::translate({8_um, 8_um});
	using fd::correction::second_order;
	using state = ins::state<dimensions>;

	auto r = grid.refinement();
	auto u = zeros(grid, domain);
	vector p{r * r, linalg::zero};
	auto ub = zeros(grid, domain);
	auto rx = bases::shape(ref, translate);
	state st = {0_s, std::move(u), std::move(p)};
	st.u[0] = {r * r, linalg::one};
	return std::make_tuple(std::move(st), std::move(ub), std::move(rx));
}

int
main(int argc, char** argv)
{
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic};
	constexpr fd::dimension y{16_um, fd::boundary::periodic};
	constexpr fd::domain domain{x, y};
	constexpr fd::mac mac{96};

	constexpr auto shear_rate = 50 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto kmin = 0.000016_s * (h / 1_um) * (h / 1_um);
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-9};

	constexpr forces::skalak1d plt_forces{1e-3_dyn/1_cm, 0*1e-1_dyn/1_cm};
	//constexpr forces::combine plt_forces{plt_tension};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", (double) params.density);

	constexpr ib::delta::roma phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	platelet1d plt{200};

	auto [st, ub, px] = initialize(mac, domain, plt, shear_rate);
	bases::container plts{plt, std::move(px)};

	ins::solver step{mac, domain, params};

	binary_writer write;
	units::time t = 0, tmax = 100_ms, k = kmin;

	auto pts = [&] (const matrix& x)
	{
		using domain_type = decltype(domain);
		constexpr auto dimensions = domain_type::dimensions;
		auto [r, c] = linalg::size(x);
		return r * c / dimensions;
	};

	auto forces_of = [&] (const auto& cell, const auto& forces,
			const units::time k, const auto& u)
	{
		const auto& ref = bases::ref(cell);
		auto w = interpolate(pts(cell.x), cell.x, u);
		auto y = cell.x + (double) k * std::move(w);
		bases::container tmp{ref, std::move(y)};
		auto g = forces(tmp);
		auto f = spread(pts(tmp.x), tmp.x, std::move(g));
		return f;
	};

	auto forces = [&] (units::time tn, const auto& v)
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
		return apply(m, zip(std::forward_as_tuple(plts),
		                    std::forward_as_tuple(plt_forces)));
	};

	auto move = [&, &st=st] (auto& cell)
	{
		auto v = interpolate(pts(cell.x), cell.x, st.u);
		cell.x += (double) k * std::move(v);
	};

	write(st, plts);
	while (t < tmax) {
		util::logging::info("simulation time: ", t);
		try { st = step(std::move(st), ub, forces); }
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return 255;
		}
		k = st.t - t;
		t = st.t;
		move(plts);
		write(st, plts);
	}
	return 0;
}
