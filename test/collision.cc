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
#include "forces/combine.h"

#include "ins/solver.h"

#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

struct binary_writer {
	units::time time = 0_s;
	units::time interval = 0.1_ms;
	std::ostream& output = std::cout;

	template <std::size_t dimensions, typename ... object_types>
	void
	operator()(const ins::state<dimensions>& state,
			bool force, const object_types& ... objects)
	{
		const auto& t = state.t;
		if (force) time = t;
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

template <typename grid_type, typename domain_type, typename ref_type>
decltype(auto)
initialize(const grid_type& grid, const domain_type& domain, const ref_type& ref)
{
	using state = ins::state<domain_type::dimensions>;
	auto r = grid.refinement();
	auto h = domain.unit() / r;
	auto u = zeros(grid, domain);
	auto ub = zeros(grid, domain);
	auto rot = bases::rotate(M_PI_2, {1, 0, 0});
	auto dx = 3.91_um * sqrt((55743711 + 7501748 * sqrt(974)) / 35) / 8750;
	auto dy = 3.91_um;

	state st = {0_s, zeros(grid, domain), vector{r*r*r, linalg::zero}};
	bases::container rbc1{ref, rot | bases::translate({8_um, 8_um, 8_um - dx - 1.5 * h})};
	bases::container rbc2{ref, rot | bases::translate({8_um, 8_um, 8_um + dx + 1.5 * h})};

	return std::make_tuple(std::move(st), std::move(ub),
			(matrix) rbc1.x, (matrix) rbc2.x);
}

struct constant {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area coeff;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		double c = coeff;
		const auto& orig = object.geometry(bases::reference).sample;
		forces::loader original{orig};

		auto size = linalg::size(orig.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__(int tid)
		{
			auto orig = original[tid];
			for (int i = 0; i < 2; ++i)
				fdata[n * i + tid] = 0;
			fdata[n * 2 + tid] = orig.s * c;
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
	constexpr fd::mac mac{96};

	constexpr auto shear_rate = 1000 / 1_s;
	constexpr auto time_scale = 1 / shear_rate;
	constexpr auto length_scale = domain.unit();
	constexpr auto h = domain.unit() / mac.refinement();
	constexpr auto kmin = 0.0000020_s * (h / 1_um) * (h / 1_um);
	constexpr ins::parameters params {kmin, time_scale, length_scale, 1_g / 1_mL, 1_cP, 1e-11};

	constexpr forces::skalak rbc_tension{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
	constexpr forces::bending rbc_bending{2e-12_erg};
	constexpr constant rbc_pos{+(1.0e-3_dyn/1_cm)};
	constexpr constant rbc_neg{-(1.0e-3_dyn/1_cm)};
	constexpr forces::combine rbc_f1{rbc_tension, rbc_bending, rbc_pos};
	constexpr forces::combine rbc_f2{rbc_tension, rbc_bending, rbc_neg};

	util::logging::info("time scale: ", params.time_scale);
	util::logging::info("length scale: ", params.length_scale);
	util::logging::info("h: ", h);
	util::logging::info("μ: ", params.viscosity);
	util::logging::info("ρ: ", params.density);

	constexpr ib::delta::bspline<2> phi;
	constexpr ib::novel::spread spread{mac, domain, phi};
	constexpr ib::novel::interpolate interpolate{mac, domain, phi};

	constexpr bases::polyharmonic_spline<7> sharp;
	rbc rbc{1600, 15000, sharp};

	auto [st, ub, r1, r2] = initialize(mac, domain, rbc);
	bases::container rbc1{rbc, std::move(r1)};
	bases::container rbc2{rbc, std::move(r2)};

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

	auto forces_of = [&] (const auto& cell, const auto& forces,
			const units::time k, const auto& u)
	{
		const auto& ref = bases::ref(cell);
		auto w = interpolate(pts(cell.x), cell.x, u);
		auto y = cell.x + (double) k * std::move(w);
		bases::container tmp{ref, std::move(y)};
		return spread(pts(tmp.x), tmp.x, forces(tmp));
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
		return apply(m, zip(std::forward_as_tuple(rbc1, rbc2),
		                    std::forward_as_tuple(rbc_f1, rbc_f2)));
	};

	auto move = [&, &st=st] (auto& cell)
	{
		auto v = interpolate(pts(cell.x), cell.x, st.u);
		cell.x += (double) k * std::move(v);
	};

	write(st, false, rbc1, rbc2);
	while (t < tmax) {
		util::logging::info("simulation time: ", t);
		try {
			st = step(std::move(st), ub, forces);
			k = st.t - t;
			t = st.t;
		}
		catch (std::runtime_error& e) {
			util::logging::error(e.what());
			write(st, true, rbc1, rbc2);
			return 255;
		}
		move(rbc1); move(rbc2);
		write(st, false, rbc1, rbc2);
	}

	return 0;
}
