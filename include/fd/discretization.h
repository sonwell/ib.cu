#include "util/sequences.h"
#include "units.h"
#include "exceptions.h"
#include "domain.h"
#include "boundary.h"


namespace fd {
namespace impl {

struct coefficients {
	using params = util::array<double, 2>;

	// Consider a boundary condition ɑu + β∂u = ɣ. Let
	// u₋₁ = Aɣ + Bu₀ be an approximation of the value at
	// ghost cell u₋₁.

	// Coefficient of the boundary condition RHS (ɣ) in the approximation
	// at the ghost point (i.e., A).
	double boundary;

	// Coefficient of the interior point (u₀) in the approximation at the
	// ghost point (i.e., B).
	double interior;

	// Coefficient of 0th-order error in approximating the 3-pt
	// Laplacian at the interior point using linear approximations
	// to u and ∂u at the boundary. (There is no 1st-order error
	// coefficient due to symmetry.)
	double laplacian;

	template <typename boundary_type>
	constexpr coefficients(const boundary_type& boundary, double h, double s) :
		coefficients(boundary.params(), h, 1 - util::math::modulo(1 - s, 1)) {}
	constexpr coefficients(const boundary::periodic& boundary, double, double) :
		boundary(0), interior(0), laplacian(0) {}
private:
	constexpr coefficients(const params& p, double h, double s) :
		coefficients(p, h, s, p[1] - p[0] * s * h) {}
	constexpr coefficients(const params& p, double h, double s, double denom) :
		boundary(-h / denom), interior((p[1] + p[0] * h * (1 - s)) / denom),
		laplacian((-(p[1] * (1 - 2 * s) - p[0] * s * h * (1 - s))) / (2 * denom)) {}
};

}

template <typename collocation, typename view_type>
class discretization : public view_type {
public:
	using view_type::solid_boundary;
	static constexpr auto on_boundary = collocation::on_boundary;
	static constexpr auto shift = collocation::shift;

	typedef struct { double lower, upper; } params;
private:
	static constexpr auto correction = solid_boundary && on_boundary;
	using weight_type = impl::coefficients;

	weight_type weights[2];

	constexpr discretization(const view_type& view, double h) :
		view_type(view), weights{{view.lower(), h, shift}, {view.upper(), -h, 1-shift}} {}
public:
	constexpr params boundary() const { return {weights[0].boundary, weights[1].boundary}; }
	constexpr params interior() const { return {weights[0].interior, weights[1].interior}; }
	constexpr params identity() const { return {weights[0].laplacian, weights[1].laplacian}; }
	constexpr auto points() const { return view_type::cells() - correction; }

	constexpr auto
	grid(double x) const
	{
		auto y = view_type::clamp(x) * view_type::resolution();
		auto z = util::math::min(points(), y - shift);
		return util::math::floor(z) + shift;
	}

	constexpr discretization(const view_type& view) :
		discretization(view, 1.0 / view.resolution()) {}
};

}
