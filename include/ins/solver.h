#pragma once
#include <utility>
#include <tuple>

#include "cuda/timer.h"
#include "algo/chebyshev.h"
#include "algo/preconditioner.h"
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/grid.h"

#include "units.h"

#include "types.h"
#include "simulation.h"
#include "diffusion.h"
#include "advection.h"
#include "boundary.h"
#include "exceptions.h"


namespace ins {

struct parameters : simulation {
	units::density density;
	units::viscosity viscosity;

	constexpr parameters(units::time k, units::time time,
			units::length length, units::density rho,
			units::viscosity mu, double tol) :
		simulation{k, time, length, mu / rho, tol},
		density(rho), viscosity(mu) {}
};

template <typename> class solver;

template <typename ... dimension_types>
class solver<fd::domain<dimension_types...>> {
private:
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
	using stepper_type = diffusion<fd::grid<domain_type>>;

	using advection_type = ins::advection<domain_type>;
	using steppers_type = std::array<stepper_type, dimensions>;
	using vectors_type = std::array<vector, dimensions>;
	using operators_type = std::array<matrix, dimensions>;

	struct poisson : solvers::mgpcg {
		virtual vector
		operator()(vector v) const
		{
			vector ones{size(v), linalg::one};
			double alpha = dot(v, ones) / ones.rows();
			util::logging::info("⟨1, ɸ⟩: ", alpha);
			if (abs(alpha) > tolerance)
				throw no_solution("⟨1, ɸ⟩ ≉ 0");
			return solvers::mgpcg::operator()(std::move(v));
		}

		template <typename grid_type>
		poisson(const grid_type& grid, double tolerance) :
			solvers::mgpcg(grid, tolerance,
					[] (const auto& g) { return fd::laplacian(g); },
					[] (const auto& g, const matrix& m) { return new mg::chebyshev(g, m); }
			) {}
	};

	template <typename lower_type, typename upper_type>
	static constexpr decltype(auto)
	solid_to_neumann(const fd::dimension<lower_type, upper_type>& dimension)
	{
		using dimension_type = fd::dimension<lower_type, upper_type>;
		if constexpr (dimension_type::solid_boundary)
			return fd::dimension{dimension.length(), fd::boundary::neumann()};
		else
			return dimension;
	}

	static constexpr decltype(auto)
	intermediate(const domain_type& domain)
	{
		using namespace util::functional;
		auto constructor = [] (auto ... components) { return fd::domain{std::move(components)...}; };
		auto transmute = [] (const auto& component) { return solid_to_neumann(component); };
		return apply(constructor, map(transmute, fd::components(domain)));
	}


	using intermediate_type = decltype(intermediate(std::declval<domain_type>()));
	using divergence_type = divergence<domain_type>;
	using gradient_type = gradient<intermediate_type>;
	using poisson_type = poisson;

	struct info_type {
		steppers_type steppers;
		operators_type operators;
		vectors_type vectors;
		poisson_type poisson;
		gradient_type gradient;
		divergence_type divergence;
	};

	template <typename tag_type>
	static info_type
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
		const auto& components = fd::components(domain);
		auto [steppers, ops, phis] = map(partial(apply, v), apply(zip, map(k, components)));

		auto interm = intermediate(domain);
		auto shifted = fd::shift::diagonally(tag);
		poisson_type poisson{fd::grid{shifted, interm}, params.tolerance};
		divergence_type div{tag, domain};
		gradient_type grad{shifted, interm};
		return {std::move(steppers), std::move(ops), std::move(phis),
			std::move(poisson), std::move(grad), std::move(div)};
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
			scal(-0.0, h);
			axpy(1 / (double) density, f, h);
			return h;
		};
		auto spmv = [&] (vector b, const matrix& op, const vector& dp)
		{
			gemv(0.5 * viscosity / density, op, dp, 1.0, b);
			return b;
		};

		double alpha = density / (frac * timestep);
		double beta = -0.5 / (frac * viscosity);
		auto h = apply(advect, u1);
		auto g = map(axpy, f, std::move(h));
		auto vb = map(spmv, ub, operators, k_grad_phi);
		auto w = map(step, steppers, u0, std::move(vb), std::move(g));
		auto divw = apply(div, w);
		auto kphi = solve(poisson, divw);
		auto p = beta * divw + alpha * kphi;
		k_grad_phi = grad(kphi);
		w = map(std::minus<void>{}, std::move(w), k_grad_phi);
		return std::make_pair(std::move(w), divw);
	}

	units::density density;
	units::viscosity viscosity;
	units::time timestep;
	units::time time_scale;
	units::length length_scale;
	advection_type advect;
	steppers_type steppers;
	vectors_type k_grad_phi;
	operators_type operators;
	poisson_type poisson;
	divergence_type div;
	gradient_type grad;

	template <typename tag_type>
	solver(const tag_type& tag, const domain_type& domain,
			const parameters& params, info_type info) :
		density(params.density),
		viscosity(params.viscosity),
		timestep(params.timestep),
		time_scale(params.time_scale),
		length_scale(params.length_scale),
		advect(tag, domain),
		steppers(std::move(info.steppers)),
		k_grad_phi(std::move(info.vectors)),
		operators(std::move(info.operators)),
		poisson(std::move(info.poisson)),
		div(std::move(info.divergence)),
		grad(std::move(info.gradient)) {}
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
		auto [v1, p1] = step(0.5, v0, vb, v0, g);
		auto [v2, p2] = step(1.0, v0, vb, v1, g);
		auto gp = grad(p2);
		return std::make_pair(redim(std::move(v2)), (double) u_scale * std::move(p2));
	}

	template <typename tag_type>
	solver(const tag_type& tag, const domain_type& domain, const parameters& params) :
		solver(tag, domain, params, construct(tag, domain, params)) {}
};

template <typename tag_type, typename grid_type>
solver(const tag_type&, const grid_type, const parameters&) ->
	solver<grid_type>;

} // namespace ins
