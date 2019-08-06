#pragma once
#include "types.h"
#include "polynomials.h"
#include "surface.h"

namespace bases {

template <int dims>
struct closed_surface : surface<dims> {
public:
	using typename surface<dims>::params;
protected:
	template <typename weight>
	static vector
	scale(const matrix& x, vector v, weight w)
	{
		auto n = x.rows();
		auto* xdata = x.values();
		auto* vdata = v.values();
		auto k = [=] __device__ (int tid, auto f)
		{
			params x;
			for (int i = 0; i < 2; ++i)
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
		//static constexpr polynomials<0> p;
		auto n = x.rows();
		//auto lu = algo::lu(fill<2>(x, phi, p));
		auto f = [=] __device__ (int tid) { return (tid < n) * 1. / n; };
		return scale(x, vector{n+1, linalg::fill(f)}, w);
		//return scale(x, solve(lu, vector{n+1, linalg::fill(f)}), w);
	}

	template <typename traits, typename basic, typename metric, typename poly>
	closed_surface(int nd, int ns, traits tr, basic phi, metric d, poly p) :
		surface<dims>(nd, ns, tr, phi, d, p) {}
};

}
