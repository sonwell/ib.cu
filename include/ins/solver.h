#pragma once
#include <utility>
#include <tuple>

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
	units::density density;
	units::viscosity viscosity;
	units::length length_scale;
	units::time time_scale;

	constexpr parameters(units::time k, units::time time,
			units::length length, units::density rho,
			units::viscosity mu, double tol) :
		simulation{k, mu / rho, tol},
		density(rho), viscosity(mu),
		length_scale(length), time_scale(time) {}
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

	using advection_type = ins::advection<domain_type>;
	using steppers_type = std::array<stepper_type, dimensions>;
	using gradient_type = std::array<vector, dimensions>;
	using operators_type = std::array<matrix, dimensions>;
	struct info_type {
		steppers_type steppers;
		operators_type operators;
		gradient_type gradient;
	};

	template <typename tag_type>
	static auto
	construct(const tag_type& tag, const domain_type& domain,
			const parameters& params)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp)
		{
			fd::grid grid{tag, domain, comp};
			diffusion stepper{grid, params};
			auto op = ins::boundary(grid, comp);
			vector v{fd::size(grid), linalg::zero};
			return std::tuple<stepper_type, matrix, vector>{
				std::move(stepper), std::move(op), std::move(v)};
		};
		auto v = [] (auto ... v) { return std::array{std::move(v)...}; };
		auto c = [] (auto ... v) { return info_type{std::move(v)...}; };
		auto objects = apply(zip, map(k, fd::components(domain)));
		return apply(c, map(partial(apply, v), std::move(objects)));
	}

	template <typename u0_type, typename ub_type, typename u1_type, typename f_type>
	decltype(auto)
	step(double frac, u0_type&& u0, ub_type&& ub, const u1_type& u1, const f_type& f)
	{
		using namespace util::functional;

		auto step = [&] (const stepper_type& stepper, const vector& u,
				vector ub, vector f)
		{
			return stepper(frac, u, std::move(ub), std::move(f));
		};
		auto axpy = [&] (const vector& f, vector h)
		{
			using linalg::axpy;
			axpy(-1 / (double) density, f, h);
			scal(-1, h);
			return h;
		};
		auto spmv = [&] (vector b, const matrix& op, const vector& dp)
		{
			gemv(frac * viscosity / density, op, dp, 1.0, b);
			return b;
		};

		auto h = apply(advect, u1);
		auto g = map(axpy, f, std::move(h));
		auto vb = map(spmv, ub, _operators, k_grad_phi);
		auto w = map(step, _steppers, u0, std::move(vb), std::move(g));
		assign(k_grad_phi, apply(projection, w));
		return w;
	}

	units::density density;
	units::viscosity viscosity;
	units::time time_scale;
	units::length length_scale;
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
			const parameters& params, info_type info) :
		density(params.density),
		viscosity(params.viscosity),
		time_scale(params.time_scale),
		length_scale(params.length_scale),
		advect(tag, domain),
		_steppers(std::move(info.steppers)),
		projection(tag, domain, params),
		k_grad_phi(std::move(info.gradient)),
		_operators(std::move(info.operators)) {}
public:
	template <typename u_type, typename ub_type, typename force_fn>
	decltype(auto)
	operator()(u_type&& u0, ub_type&& ub, const force_fn& forces)
	{
		using namespace util::functional;
		static constexpr auto scalem = [] (double mu)
		{
			return partial(map, [=] (vector v) { return mu * std::move(v); });
		};

		auto u_scale = length_scale / time_scale;
		auto nondim = scalem(1.0 / u_scale);
		auto redim = scalem(u_scale);
		auto half = scalem(0.5);

		auto g = nondim(forces(half(u0)));
		auto v0 = nondim(std::forward<u_type>(u0));
		auto vb = nondim(std::forward<ub_type>(ub));
		auto v1 = step(0.5, v0, vb, v0, g);
		v1 = step(1.0, v0, vb, v1, g);
		return redim(std::move(v1));
	}

	template <typename tag_type>
	solver(const tag_type& tag, const domain_type& domain, const parameters& params) :
		solver(tag, domain, params, construct(tag, domain, params)) {}
};

template <typename tag_type, typename grid_type>
solver(const tag_type&, const grid_type, const parameters&) ->
	solver<grid_type>;

} // namespace __1

using __1::solver;

} // namespace ins
