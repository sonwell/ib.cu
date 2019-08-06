#pragma once
#include "bases/types.h"
#include "bases/shapes/sphere.h"
#include "bases/traits.h"
#include "bases/polynomials.h"
#include "units.h"

struct rbc : bases::shapes::sphere {
private:
	using matrix = bases::matrix;
	static constexpr bases::traits<rbc> traits;
	static constexpr bases::polynomials<0> p;
public:
	static matrix
	shape(const matrix& params)
	{
		static constexpr double radius = 3.91_um;
		static constexpr double shift = 25_um;
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto y = cos(t) * cos(p);
			auto x = sin(t) * cos(p);
			auto z0 = sin(p);
			auto r2 = y*y + x*x;
			auto z = 0.5 * z0 * (0.21 + 2.0 * r2 - 1.12 * r2*r2);

			xdata[0 * rows + tid] = shift + radius * x;
			xdata[1 * rows + tid] = shift + radius * y;
			xdata[2 * rows + tid] = shift + radius * z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <typename basic>
	rbc(int nd, int ns, basic phi) :
		bases::shapes::sphere(nd, ns, traits, phi, p) {}
};
