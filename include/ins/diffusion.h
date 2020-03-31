#pragma once
#include <utility>
#include "fd/identity.h"
#include "fd/laplacian.h"
#include "fd/correction.h"
#include "mg/chebyshev.h"
#include "util/log.h"
#include "types.h"
#include "simulation.h"
#include "solvers.h"

namespace ins {

template <typename grid_type>
class diffusion {
public:
	using parameters = simulation;
	static constexpr auto dimensions = grid_type::dimensions;
private:
	static constexpr auto helmholtz = [] (units::unit<2, 0, 0> l, const auto& g)
	{
		using fd::correction::second_order;
		auto id = fd::identity(g, second_order);
		auto lh = fd::laplacian(g);
		axpy(-(double) l, lh, id);
		return id;
	};

	static units::unit<2, 0, 0>
	lambda(const parameters& p)
	{
		return p.timestep * p.coefficient / 2;
	}

	struct chebyshev : solvers::chebpcg {
		chebyshev(const grid_type& grid, double tolerance, units::unit<2, 0, 0> l) :
			chebpcg(tolerance, helmholtz(l, grid)) {}

		chebyshev(const grid_type& grid, const parameters& p) :
			chebyshev(grid, p.tolerance, lambda(p)) {}
	};

	struct multigrid : solvers::mgpcg {
		static constexpr auto smoother = [] (const auto& g, const matrix& m)
		{
			return new mg::chebyshev(g, m);
		};

		multigrid(const grid_type& grid, double tolerance, units::unit<2, 0, 0> l) :
			mgpcg(grid, tolerance, [=] (auto&& g) { return helmholtz(l, g); }, smoother) {}

		multigrid(const grid_type& grid, const parameters& p):
			multigrid(grid, p.tolerance, lambda(p)) {}
	};

	units::time timestep;
	units::diffusivity coefficient;
	double tolerance;
	matrix identity;
	matrix laplacian;
	chebyshev solver;
public:
	vector
	operator()(double frac, const vector& u, vector rhs, const vector& f) const
	{
		double mu = coefficient;
		double k = frac * timestep;
		gemv(1.0, laplacian, u, 1.0, rhs);
		gemv(k, identity, f, k * mu, rhs);
		util::logging::info("helmholtz solve ", abs(rhs) / frac);
		return solve(solver, std::move(rhs)) + u;
	}

	diffusion(const grid_type& grid, const parameters& params) :
		timestep(params.timestep),
		coefficient(params.coefficient),
		identity(fd::identity(grid, fd::correction::second_order)),
		laplacian(fd::laplacian(grid)),
		solver(grid, params) {}

	diffusion(diffusion&& other) :
		timestep(other.timestep),
		coefficient(other.coefficient),
		identity(std::move(other.identity)),
		laplacian(std::move(other.laplacian)),
		solver(std::move(other.solver)) {}
};

template <typename grid_type>
diffusion(const grid_type&, const simulation&)
	-> diffusion<grid_type>;

} // namespace ins
