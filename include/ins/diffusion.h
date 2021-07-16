#pragma once
#include <utility>
#include "cuda/timer.h"
#include "fd/identity.h"
#include "fd/laplacian.h"
#include "fd/correction.h"
#include "solvers/chebpcg.h"
#include "util/log.h"
#include "types.h"
#include "simulation.h"

namespace ins {
// Implicitly solve diffusion equation by writing discretization as
//
//     (Ĩ-½λΔₕ)δu = ɣλΔₕu + kĨf.
//
class diffusion {
public:
	using parameters = simulation;
private:
	static double
	lambda(units::time k, units::diffusivity d)
	{
		return k * d / 2;
	}

	void
	set_timestep(const units::time& step)
	{
		if (step == current_timestep)
			return;
		auto l = lambda(step, coefficient);
		helmholtz = {identity - l * laplacian, tolerance};
		current_timestep = step;
	}

	units::time current_timestep;
	units::diffusivity coefficient;
	double tolerance;
	matrix identity;
	matrix laplacian;
	solvers::chebpcg helmholtz;
public:
	util::getset<units::time> timestep;

	vector
	operator()(double gamma, const vector& u, vector rhs, const vector& f) const
	{
		// gamma: 0.5 => backward Euler
		//        1.0 =? Crank-Nicolson
		double mu = coefficient;
		double k = gamma * timestep;
		gemv(1.0, laplacian, u, 1.0, rhs);
		gemv(k, identity, f, k * mu, rhs);
		util::logging::debug("helmholtz solve ", abs(rhs) / gamma);
		return solve(helmholtz, std::move(rhs)) + u;
	}

	template <typename grid_type>
	diffusion(const grid_type& grid, const parameters& params) :
		current_timestep(params.timestep),
		coefficient(params.coefficient),
		tolerance(params.tolerance),
		identity(fd::identity(grid, fd::correction::second_order)),
		laplacian(fd::laplacian(grid)),
		helmholtz(identity - lambda(params.timestep, params.coefficient) * laplacian, tolerance),
		timestep([&] () -> decltype(auto) { return current_timestep; },
		         [&] (const units::time& step) { set_timestep(step); }) {}
	diffusion(diffusion&& other) :
		current_timestep(other.current_timestep),
		coefficient(other.coefficient),
		tolerance(other.tolerance),
		identity(std::move(other.identity)),
		laplacian(std::move(other.laplacian)),
		helmholtz(std::move(other.helmholtz)),
		timestep([&] () -> decltype(auto) { return current_timestep; },
		         [&] (const units::time& step) { set_timestep(step); }) {}
};

} // namespace ins
