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

#include "forces/skalak.h"
#include "forces/bending.h"
#include "forces/combine.h"

#include "ins/solver.h"

#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

struct state {
	units::time t;
	vector u, v, w;
	vector p;
	matrix r1, r2;
};

template <typename op_type>
inline void
io(op_type op)
{
	using namespace util::functional;
	std::tuple fns{
		[] (auto&& st) -> decltype(auto) { return st.t; },
		[] (auto&& st) -> decltype(auto) { return st.u; },
		[] (auto&& st) -> decltype(auto) { return st.v; },
		[] (auto&& st) -> decltype(auto) { return st.w; },
		[] (auto&& st) -> decltype(auto) { return st.p; },
		[] (auto&& st) -> decltype(auto) { return st.r1; },
		[] (auto&& st) -> decltype(auto) { return st.r2; }
	};
	map([&] (auto f) { op(f); }, fns);
}


std::ostream&
operator<<(std::ostream& out, const state& st)
{
	io([&] (auto f) { out << linalg::io::binary << f(st); });
	return out;
}

std::istream&
operator>>(std::istream& in, state& st)
{
	io([&] (auto f) { in >> linalg::io::binary >> f(st); });
	return in;
}

struct binary_writer {
	static constexpr int steps_per_print = 1000;
	std::ostream& output = std::cout;
	int count = 0;

	template <typename u_type>
	void
	operator()(units::time t, const u_type& vel, const vector& p,
			const matrix& x, const matrix& y, bool force = false)
	{
		if ((count++) % steps_per_print && !force) return;
		auto&& [u, v, w] = vel;
		state st{t, u, v, w, p, x};
		operator()(st);
	}

	void operator()(const state& st) { output << st; }

	binary_writer(std::ostream& output = std::cout) :
		output(output) {}
};

struct null_writer {
	null_writer(std::ostream& = std::cout) {}

	template <typename u_type>
	void operator()(const u_type&, const vector&, const matrix&, bool = false) {}
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
	state st;
	auto r = grid.refinement();
	auto ub = zeros(grid, domain);
	auto rot = bases::rotate(1, {1, 0, 0});
	bases::container rbc1{ref, rot | bases::translate({8_um, 8_um, 6_um})};
	bases::container rbc2{ref, rot | bases::translate({8_um, 8_um, 10_um})};

	st.t = 0;
	std::tie(st.u, st.v, st.w) = zeros(grid, domain);
	st.p = vector{r * r * r, linalg::zero};
	st.r1 = (matrix) rbc1.x;
	st.r2 = (matrix) rbc2.x;
	return std::make_pair(std::move(st), std::move(ub));
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
	rbc rbc{1200, 7500, sharp};

	auto [st, ub] = initialize(mac, domain, rbc);
	units::time t = st.t;
	std::tuple u = {std::move(st.u), std::move(st.v), std::move(st.w)};
	vector p = std::move(st.p);
	bases::container rbc1{rbc, std::move(st.r1)};
	bases::container rbc2{rbc, std::move(st.r2)};

	ins::solver step{mac, domain, params};

	binary_writer write;
	units::time tmax = 1_ms, k = kmin;

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
		return apply(m, zip(std::forward_as_tuple(rbc1, rbc2),
		                    std::forward_as_tuple(rbc_f1, rbc_f2)));
	};

	int err = 0;
	write(t, u, p, rbc1.x, rbc2.x);
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
		m(rbc1); m(rbc2);
		write(t, u, p, rbc1.x, rbc2.x);
	}
	write(t, u, p, rbc1.x, rbc2.x, true);

	return err;
}
