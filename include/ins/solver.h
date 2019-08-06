#pragma once
#include <utility>

#include "algo/chebyshev.h"
#include "algo/preconditioner.h"
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/grid.h"

#include "units.h"

#include "types.h"
#include "simulation.h"
#include "projection.h"
#include "diffusion.h"
#include "advection.h"
#include "boundary.h"


namespace ins {

struct parameters : simulation {
public:
	units::density density;
	units::viscosity viscosity;

	constexpr parameters(units::time k, units::density rho, units::viscosity mu, double tol) :
		simulation{k, mu / rho, tol}, density(rho), viscosity(mu) {}
};

namespace __1 {

template <typename> class solver;

template <typename ... dimension_types>
class solver<fd::domain<dimension_types...>> {
private:
	template <typename T, typename R> using replace = R;
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
	using projection_type = ins::projection<domain_type>;
	using stepper_type = diffusion<fd::grid<domain_type>>;
	using grids_type = std::tuple<replace<dimension_types, fd::grid<domain_type>>...>;

	template <typename tag_type>
	static auto
	grids(const tag_type& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& view)
		{
			return fd::grid{tag, domain, view};
		};
		return map(k, fd::components(domain));
	}

	static auto
	steppers(const grids_type& grids, const parameters& params)
	{
		using namespace util::functional;
		auto k = [&] (const auto& grid)
		{
			return diffusion{grid, params};
		};
		return map(k, grids);
	}

	static auto
	operators(const grids_type& grids)
	{
		using namespace util::functional;
		auto k = [&] (const auto& grid) { return ins::boundary(grid); };
		return map(k, grids);
	}

	auto
	initialize(const grids_type& grids)
	{
		using namespace util::functional;
		auto k = [&] (const auto& grid)
		{
			return vector{fd::size(grid), algo::fill(0.0)};
		};
		return map(k, grids);
	}

	template <typename u0_type, typename ub_type, typename u1_type, typename force_fn>
	decltype(auto)
	step(double frac, u0_type&& u0, ub_type&& ub, const u1_type& u1, const force_fn& forces)
	{
		using namespace util::functional;

		auto step = [&] (const stepper_type& stepper, const vector& u,
				const vector& ub, const vector& f)
		{
			return stepper(frac, u, ub, f);
		};
		auto axpy = [&] (vector f, const vector& h) { return std::move(f) / (double) density - h; };
		auto spmv = [&] (vector b, const matrix& op, const vector& dp)
		{
			gemv(frac * viscosity / density, op, dp, 1.0, b);
			return b;
		};

		auto h = apply(advect, u1);
		auto v = map([&] (const vector& u) { return frac * u; }, u1);
		auto f = map(axpy, forces(std::move(v)), h);
		auto vb = map(spmv, ub, _operators, k_grad_phi);
		auto w = map(step, _steppers, u0, vb, f);
		auto p = apply(projection, w);
		return std::pair{std::move(w), std::move(p)};
	}

	using advection_type = ins::advection<domain_type>;
	using steppers_type = std::tuple<replace<dimension_types, stepper_type>...>;
	using gradient_type = std::tuple<replace<dimension_types, vector>...>;
	using operators_type = std::tuple<replace<dimension_types, matrix>...>;

	units::density density;
	units::viscosity viscosity;
	advection_type advect;
	steppers_type _steppers;
	projection_type projection;
	gradient_type k_grad_phi;
	operators_type _operators;

	template <typename to_type, typename from_type>
	static constexpr decltype(auto)
	assign(to_type&& to, from_type&& from)
	{
		using namespace util::functional;
		auto k = [] (auto& t, auto&& f) { t = std::forward<decltype(f)>(f); };
		map(k, to, std::forward<from_type>(from));
		return to;
	}

	template <typename tag_type>
	solver(const tag_type& tag, const domain_type& domain,
			const grids_type& grids, const parameters& params) :
		density(params.density),
		viscosity(params.viscosity),
		advect(tag, domain),
		_steppers(steppers(grids, params)),
		projection(tag, domain, params),
		k_grad_phi(initialize(grids)),
		_operators(operators(grids)) {}
public:
	template <typename u_type, typename ub_type, typename force_fn>
	decltype(auto)
	operator()(u_type&& u0, ub_type&& ub, const force_fn& forces)
	{
		using namespace util::functional;
		auto [u1, gp1] = step(0.5, u0, ub, u0, forces);
		auto [u2, gp2] = step(1.0, u0, ub, u1, forces);
		assign(k_grad_phi, std::move(gp2));
		return u2;
	}

	template <typename tag_type>
	solver(const tag_type& tag, const domain_type& domain, const parameters& params) :
		solver(tag, domain, grids(tag, domain), params) {}
};

template <typename tag_type, typename grid_type>
solver(const tag_type&, const grid_type, const parameters&) ->
	solver<grid_type>;

} // namespace __1

using __1::solver;

} // namespace ins
