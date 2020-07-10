#pragma once
#include "types.h"
#include "polynomials.h"
#include "phs.h"
#include "surface.h"

namespace bases {

template <int dims>
struct closed_surface : surface<dims> {
protected:
	using surface<dims>::shape;

	template <typename weight>
	static vector
	scale(const matrix& x, vector v, weight w)
	{
		// Given integration weights `v` for a homogeneous surface with surface
		// area 1, discretized in parameter space at points, `x`, compute the
		// integration weights in parameter space. All of the relevant info is
		// in the function `weight` and is specific to a particular surface.
		auto n = x.rows();
		auto* xdata = x.values();
		auto* vdata = v.values();
		auto k = [=] __device__ (int tid, auto f)
		{
			double x[dims];
			for (int i = 0; i < dims; ++i)
				x[i] = xdata[i * n + tid];
			vdata[tid] /= f(x);
		};
		util::transform<128, 8>(k, n, w);
		return v;
	}

	template <typename metric, typename weight,
	          typename = std::enable_if_t<is_metric_v<metric>>>
	static vector
	weights(const matrix& x, metric distance, weight w)
	{
		// Compute surface integration weights.
		// The surface must be "homogeneous", i.e., ∫ ɸ(x(θ)-x0) dθ over the
		// surface must be independent of x0. Then, for p_0(x) = 1,
		//
		//     c^i ɸ(x_j-x_i) ≈ ∫ ɸ(x(θ)-x_i) dθ
		//       c^i p_0(x_i) = σ,
		//
		// where σ is the surface area of the surface. Let d = -∫ ɸ(x(θ)-x0) dθ
		// be unknown exactly (or at least, we don't need to know it). Then, we
		// solve
		//
		//     [  Φ   1 ][c⃗] = [0⃗] = σ[0⃗]
		//     [ 1^T  0 ][d]   [σ]    [1]
		//
		// The system is solved with a surface area of 1 and we scale the result
		// to get correct results, if necessary.  To get dθ, divide c^i by the
		// Jacobian |∂x_i/∂θ|. For any topologically equivalent surface,
		// integration weights are found from dθ and the Jacobian of the new
		// surface.
		constexpr polyharmonic_spline<1> basic;
		constexpr polynomials<0> p;
		rbf phi{basic, distance};
		auto n = x.rows();
		auto lu = algo::lu(fill<dims>(x, phi, p));
		auto f = [=] __device__ (int tid) { return tid >= n; };
		return scale(x, solve(lu, vector{n+1, linalg::fill(f)}), w);
	}

	template <typename traits_type, typename interp, typename eval, typename metric, typename poly,
	          typename = std::enable_if_t<is_basic_function_v<interp> &&
	                                      is_basic_function_v<eval>   &&
	                                      is_metric_v<metric>         &&
	                                      is_polynomial_basis_v<poly>>>
	closed_surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, metric d, poly p) :
		surface<dims>(nd, ns, tr, phi, psi, d, p) {}

	template <typename traits_type, typename basic, typename metric, typename poly>
	closed_surface(int nd, int ns, traits<traits_type> tr, basic phi, metric d, poly p) :
		closed_surface(nd, ns, tr, phi, phi, d, p) {}
};

}
