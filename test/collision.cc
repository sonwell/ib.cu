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

#include "forces/skalak.h"
#include "forces/bending.h"
#include "forces/dissipation.h"
#include "forces/combine.h"

#include "ins/solver.h"
#include "ins/state.h"

#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.01_ms;
	std::ostream& output = std::cout;

	template <std::size_t dimensions>
	void
	operator()(const ins::state<dimensions>& state,
			const matrix& x, const matrix& f)
	{
		const auto& t = state.t;
		if (t < time) return;
		time += interval;

		output << linalg::io::binary << state.t;
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

template <typename grid_type, typename domain_type, typename ref_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain, const ref_type& ref)
{
	using state = ins::state<domain_type::dimensions>;
	auto r = grid.refinement();
	auto u = zeros(grid, domain);
	auto rot1 = bases::rotate(M_PI_2, {-1, 0, 1});
	auto rot2 = bases::rotate(M_PI_2, {-1, 0, 1});

	state st = {0_s, zeros(grid, domain), vector{r*r*r, linalg::zero}};
	auto rbc = bases::shape(ref,
			rot1 | bases::translate({4_um, 8_um, 4_um}),
			rot2 | bases::translate({12_um, 8_um, 12_um})
	);

	return std::make_tuple(std::move(st), std::move(rbc));
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
	constexpr fd::mac mac{80};

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
	constexpr forces::combine rbc_forces{rbc_tension, rbc_bending, rbc_dissip};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);

	constexpr ib::delta::bspline<3> phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> sharp;
	rbc rbc{2500, 10000, sharp};

	auto [st, r] = initialize(mac, domain, rbc);
	auto ub = zeros(mac, domain);
	bases::container rbcs{rbc, std::move(r)};

	ins::solver step{mac, domain, params};

	binary_writer write;
	units::time t = st.t, tmax=10_ms, k = kmin;

	auto pts = [&] (const matrix& x)
	{
		using domain_type = decltype(domain);
		constexpr auto dimensions = domain_type::dimensions;
		auto [r, c] = linalg::size(x);
		return r * c / dimensions;
	};

	matrix f;
	auto forces_of = [&] (const auto& cell, const auto& forces,
			const units::time k, const auto& u)
	{
		const auto& ref = bases::ref(cell);
		auto w = interpolate(pts(cell.x), cell.x, u);
		auto y = cell.x + (double) k * w;
		bases::container tmp{ref, std::move(y)};
		f = forces(tmp, w);
		return spread(pts(tmp.x), tmp.x, f);
	};

	auto forces = [&] (units::time tn, const auto& v)
	{
		using namespace util::functional;
		static constexpr std::plus<vector> plus;
		units::time k = tn - t;
		auto f = [&] (const auto& pair)
		{
			auto& [obj, fn] = pair;
			return forces_of(obj, fn, k, v);
		};
		auto op = [&] (auto l, auto r)
			{ return map(plus, std::move(l), f(r)); };
		auto m = [&] (const auto& head, const auto& ... tail)
			{ return foldl(op, f(head), tail...); };
		return apply(m, zip(std::forward_as_tuple(rbcs),
		                    std::forward_as_tuple(rbc_forces)));
	};

	auto move = [&, &st=st] (auto& cell)
	{
		auto v = interpolate(pts(cell.x), cell.x, st.u);
		cell.x += (double) k * std::move(v);
	};

	forces(t, st.u);
	write(st, rbcs.x, f);
	while (t < tmax) {
		util::logging::info("simulation time: ", t);
		try { st = step(std::move(st), ub, forces); }
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			return 255;
		}
		k = st.t - t;
		t = st.t;
		move(rbcs);
		write(st, rbcs.x, f);
	}

	return 0;
}
