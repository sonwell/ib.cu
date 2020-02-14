#pragma once
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/domain.h"
#include "fd/laplacian.h"

#include "algo/pcg.h"
#include "algo/preconditioner.h"

#include "mg/solver.h"
#include "mg/chebyshev.h"

#include "util/functional.h"
#include "util/log.h"

#include "types.h"
#include "differential.h"
#include "solvers.h"
#include "simulation.h"
#include "exceptions.h"

namespace ins {
namespace __1 {

template <typename lower_type, typename upper_type>
constexpr decltype(auto)
dimension(const fd::dimension<lower_type, upper_type>& dimension)
{
	if constexpr (std::is_same_v<lower_type, fd::boundary::periodic>)
		return dimension;
	else
		return fd::dimension{dimension.length(), fd::boundary::neumann()};
}

template <typename ... dimension_types>
constexpr decltype(auto)
intermediate(const fd::domain<dimension_types...>& domain)
{
	using namespace util::functional;
	auto k = [] (auto ... components) { return fd::domain{std::move(components)...}; };
	auto c = [] (const auto& component) { return dimension(component); };
	return apply(k, map(c, fd::components(domain)));
}

} // namespace __1

template <typename> class projection;

template <typename ... dimension_types>
class projection<fd::domain<dimension_types...>> {
public:
	using parameters = simulation;
private:
	struct multigrid : solvers::mgpcg {
		template <typename grid_type>
		multigrid(const grid_type& grid, double tolerance) :
			solvers::mgpcg(grid, tolerance,
					[] (const auto& g) { return fd::laplacian(g); },
					[] (const auto& g, const matrix& m) { return new mg::chebyshev(g, m); }
			) {}
	};

	using domain_type = fd::domain<dimension_types...>;
	using interm_domain_type = decltype(__1::intermediate(std::declval<domain_type>()));
	using divergence_functor_type = divergence<domain_type>;
	using gradient_functor_type = gradient<interm_domain_type>;
	using multigrid_solver_type = multigrid;

	double tolerance;
	double length_scale;
	multigrid_solver_type solver;
	divergence_functor_type div;
	gradient_functor_type grad;

	template <typename tag_type>
	static constexpr decltype(auto)
	shift(const tag_type& tag)
	{
		return fd::shift::diagonally(tag);
	}

	template <typename tag_type, typename domain_type, typename shifted_tag_type, typename interm_type>
	projection(const tag_type& tag, const domain_type& domain, const shifted_tag_type& stag,
			const interm_type& interm, const parameters& params) :
		tolerance(params.tolerance),
		length_scale(params.length_scale),
		solver(fd::grid{stag, interm}, tolerance),
		div(tag, domain),
		grad(stag, interm) {}
public:
	static constexpr auto dimensions = domain_type::dimensions;

	template <typename ... vector_types>
	auto
	operator()(vector_types& ... vectors)
	{
		static_assert(sizeof...(vector_types) == dimensions,
				"number of supplied vectors matches dimensions");
		using namespace util::functional;
		auto div_u = div(vectors...);
		// project out nullspace
		vector ones{size(div_u), linalg::one};
		double alpha = dot(div_u, ones) / ones.rows();
		if (abs(alpha) > tolerance)
			throw no_solution("⟨1, ∇·u*⟩ ≉ 0");
		util::logging::info("⟨1, ∇·u*⟩: ", alpha);
		axpy(-alpha, ones, div_u);
		// solve kΔϕ = ∇·u
		auto dt_phi = solve(solver, length_scale * std::move(div_u));
		auto dt_grad_phi = grad(std::move(dt_phi) / length_scale);
		// update u := u - k∇ϕ
		auto subtract = [] (auto& l, const auto& r) { l -= r; };
		map(subtract, std::tie(vectors...), dt_grad_phi);
		return dt_grad_phi;
	}

	template <typename tag_type>
	projection(const tag_type& tag, const domain_type& domain, const parameters& params) :
		projection(tag, domain, shift(tag), __1::intermediate(domain), params) {}
};

template <typename tag_type, typename domain_type>
projection(const tag_type&, const domain_type&, const simulation&) ->
	projection<domain_type>;

} // namespace ins
