#pragma once
#include "types.h"
#include "polynomials.h"
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

	template <typename rbf, typename weight>
	static vector
	weights(const matrix& x, rbf phi, weight w)
	{
		constexpr polynomials<0> p;
		auto n = x.rows();
		auto lu = algo::lu(fill<dims>(x, phi, p));
		auto f = [=] __device__ (int tid) { return tid >= n; };
		return scale(x, solve(lu, vector{n+1, linalg::fill(f)}), w);
	}

	template <typename traits_type, typename interp, typename eval, typename metric, typename poly>
	closed_surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, metric d, poly p) :
		surface<dims>(nd, ns, tr, phi, psi, d, p) {}

	template <typename traits_type, typename basic, typename metric, typename poly>
	closed_surface(int nd, int ns, traits<traits_type> tr, basic phi, metric d, poly p) :
		closed_surface(nd, ns, tr, phi, phi, d, p) {}
};

}
