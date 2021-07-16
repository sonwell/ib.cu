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
#include "ib/solver.h"
#include "cuda/event.h"
#include "units.h"
#include "platelet.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.1_ms;
	std::ostream& output = std::cout;

	template <std::size_t dimensions, std::size_t n>
	void
	operator()(const ib::state<dimensions, n>& state, const matrix& x, const matrix& f)
	{
		if (state.t < time) return;
		time += interval;

		output << linalg::io::binary << state;
		output << linalg::io::binary << x;
		output << linalg::io::binary << f;
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
	auto translate = bases::shear({{1.1, 0.0}, {0.0, 1.0/1.1}}) | bases::translate({8_um, 8_um});
	using fd::correction::second_order;
	using state = ib::state<dimensions, 1>;

	auto r = grid.refinement();
	auto u = zeros(grid, domain);
	vector p{r * r, linalg::zero};
	auto ub = zeros(grid, domain);
	auto rx = bases::shape(ref, translate);
	state st = {0_s, std::move(u), std::move(p), std::array{std::move(rx)}};
	return std::pair{std::move(st), std::move(ub)};
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
	constexpr auto kmin = 1e-6_s;
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-9};

	constexpr forces::skalak1d plt_forces{1e-3_dyn/1_cm, 0*1e-1_dyn/1_cm};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", (double) params.density);


	constexpr bases::polyharmonic_spline<7> sharp;
	platelet1d plt{100, 400, sharp};

	constexpr ib::delta::roma phi;
	ib::solver step{mac, domain, phi, params};

	auto [st, ub] = initialize(mac, domain, plt, shear_rate);

	binary_writer write{st.t};
	units::time tmax = 1_ms;

	matrix py, pf;
	auto forces = [&] (units::time t, const auto& x, const auto& u)
	{
		const auto& [px] = x;
		const auto& [pu] = u;

		const bases::container plts{plt, px};
		py = plts.geometry(bases::current).sample.position;
		pf = plt_forces(plts, pu);

		return std::pair{std::array{py}, std::array{pf}};
	};

	forces(st.t, st.x, st.x);
	write(st, py, pf);
	while (st.t < tmax) {
		util::logging::info("simulation time: ", st.t);
		try { st = step(std::move(st), ub, forces); }
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return 255;
		}
		write(st, py, pf);
	}
	return 0;
}
