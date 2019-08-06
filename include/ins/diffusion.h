#pragma once
#include <memory>
#include <utility>
#include <array>
#include <future>

#include "fd/identity.h"
#include "fd/laplacian.h"
#include "fd/correction.h"
#include "util/functional.h"
#include "util/log.h"
#include "algo/preconditioner.h"
#include "algo/chebyshev.h"
#include "algo/symmilu.h"
#include "algo/redblack.h"
#include "algo/pcg.h"
#include "types.h"
#include "simulation.h"

namespace ins {
namespace __1 {

struct chebyshev : algo::chebyshev, algo::preconditioner {
private:
	chebyshev(std::pair<double, double> range, const algo::matrix& m) :
		algo::chebyshev(std::get<1>(range), std::get<0>(range), m) {}
public:
	virtual algo::vector
	operator()(const algo::vector& b) const
	{
		return algo::chebyshev::operator()(b);
	}

	chebyshev(const algo::matrix& m) :
		chebyshev(algo::gershgorin(m), m) {}
};

struct plu : algo::symmilu {
	template <typename grid_type>
	plu(const grid_type& grid, const algo::matrix& m) :
		algo::symmilu{m, new algo::redblack{grid}} {}
};

template <typename grid_type>
class diffusion {
public:
	using parameters = simulation;
	static constexpr auto dimensions = grid_type::dimensions;
private:
	static constexpr auto& order = fd::correction::second_order;
	using pointer = std::unique_ptr<algo::preconditioner>;

	static decltype(auto)
	construct(const parameters& params, matrix identity, const matrix& laplacian)
	{
		double lambda = params.timestep * params.coefficient / 2;
		axpy(-lambda, laplacian, identity);
		return identity;
	}

	units::time timestep;
	units::diffusivity coefficient;
	double tolerance;
	matrix identity;
	matrix laplacian;
	matrix helmholtz;
	chebyshev preconditioner;
public:
	vector
	operator()(double frac, const vector& u, vector rhs,
			const vector& f) const
	{
		using algo::krylov::pcg;
		double scale = 1.0; //1_m / 1_s;
		double mu = coefficient;
		double k = frac * (double) timestep / scale;
		gemv(1.0, identity, f, mu, rhs);
		gemv(k * mu, laplacian, u, k, rhs);
		util::logging::info("helmholtz solve ", abs(rhs) / frac);
		return scale * pcg(preconditioner, helmholtz, rhs, tolerance) + u;
	}

	diffusion(const grid_type& grid, const parameters& params) :
		timestep(params.timestep),
		coefficient(params.coefficient),
		tolerance(params.tolerance),
		identity(fd::identity(grid, order)),
		laplacian(fd::laplacian(grid)),
		helmholtz(construct(params, identity, laplacian)),
		preconditioner(helmholtz) {}

	diffusion(diffusion&& other) :
		timestep(other.timestep),
		coefficient(other.coefficient),
		tolerance(other.tolerance),
		identity(std::move(other.identity)),
		laplacian(std::move(other.laplacian)),
		helmholtz(std::move(other.helmholtz)),
		preconditioner(helmholtz) {}
};

template <typename grid_type>
diffusion(const grid_type&, const simulation&)
	-> diffusion<grid_type>;

} // namespace __1

using __1::diffusion;

} // namespace ins
