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
#include "bases/phs.h"
#include "bases/scaled.h"
#include "bases/polynomials.h"
#include "bases/geometry.h"

#include "ib/novel.h"
#include "ib/bspline.h"
#include "ib/cosine.h"
#include "ib/solver.h"
#include "ib/state.h"

#include "forces/skalak.h"
#include "forces/bending.h"
#include "forces/dissipation.h"
#include "forces/combine.h"

#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.01_ms;
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

template <typename grid_type, typename domain_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain, const rbc& ref)
{
	using state = ib::state<domain_type::dimensions, 1>;
	auto r = grid.refinement();
	auto u = zeros(grid, domain);
	auto rot1 = bases::rotate(M_PI_2, {-1, 0, 1});
	auto rot2 = bases::rotate(M_PI_2, {-1, 0, 1});

	auto rbc = bases::shape(ref,
			rot1 | bases::translate({4_um, 8_um, 4_um}),
			rot2 | bases::translate({12_um, 8_um, 12_um})
	);
	vector p{r*r*r, linalg::zero};
	return state{0_s, zeros(grid, domain), std::move(p),
	             std::array{std::move(rbc)}};
}

struct constant {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area coeff;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object, const matrix&) const
	{
		constexpr auto angle = M_PI_4;
		double c = coeff;
		const auto& curr = object.geometry(bases::current).sample;
		const auto& orig = object.geometry(bases::reference).sample;
		forces::loader<forces::measure> area{orig};

		auto size = linalg::size(curr.position);
		auto m = size.rows;
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__(int tid)
		{
			auto s = area[tid];
			auto sign = 1 - 2 * ((tid / m) & 1);
			fdata[n * 0 + tid] = sign * sin(angle) * s * c;
			fdata[n * 1 + tid] = 0.0;
			fdata[n * 2 + tid] = sign * cos(angle) * s * c;
		};
		util::transform(k, n);
		return f;
	}

	constexpr constant(const energy_per_area coeff) : coeff{coeff} {}
};

int
main(int argc, char** argv)
{
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{16_um, fd::boundary::periodic};
	constexpr fd::dimension y{16_um, fd::boundary::dirichlet};
	constexpr fd::dimension z{16_um, fd::boundary::periodic};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{20};

	constexpr auto shear_rate = 1000 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto kmin = 50_ns;
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-11};

	constexpr forces::skalak rbc_tension{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
	constexpr forces::bending rbc_bending{2e-12_erg};
	constexpr forces::dissipation rbc_dissip{2.5e-7_dyn*1_s/1_cm};
	constexpr constant rbc_constant{1e-3_dyn/1_cm};
	constexpr forces::combine rbc_forces{rbc_tension, rbc_bending,
	                                     rbc_dissip, rbc_constant};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);

	constexpr ib::delta::bspline<3> phi;
	//constexpr ib::novel::spread spread{mac, domain, phi};
	//constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> sharp;
	rbc rbc{1000, 1000, sharp};

	auto st = initialize(mac, domain, rbc);
	auto ub = zeros(mac, domain);
	//bases::container rbcs{rbc, std::move(r)};

	ib::solver step{mac, domain, phi, params};

	binary_writer write;
	units::time tmax = 10_ms;

	auto forces = [&] (units::time t, const auto& x, const auto& u)
	{
		auto& [rx] = x;
		auto& [ru] = u;

		bases::container rbcs{rbc, rx};
		auto rf = rbc_forces(rbcs, ru);
		auto ry = rbcs.geometry(bases::current).sample.position;

		return std::pair{std::array{std::move(ry)},
		                 std::array{std::move(rf)}};
	};

	write(st);
	while (st.t < tmax) {
		util::logging::info("simulation time: ", st.t);
		try { st = step(std::move(st), ub, forces); }
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return 255;
		}
		write(st);
	}

	return 0;
}
