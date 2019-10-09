#pragma once
#include "bases/types.h"
#include "bases/shapes/sphere.h"
#include "bases/traits.h"
#include "bases/polynomials.h"
#include "units.h"

struct platelet : bases::shapes::sphere {
private:
	using matrix = bases::matrix;
	static constexpr bases::traits<platelet> traits;
	static constexpr bases::polynomials<0> p;
public:
	static matrix
	shape(const matrix& params)
	{
		static constexpr double major = 1.29_um;
		static constexpr double minor = 0.43_um;
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z = sin(p);

			xdata[0 * rows + tid] = major * x;
			xdata[1 * rows + tid] = minor * y;
			xdata[2 * rows + tid] = major * z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <typename basic>
	platelet(int nd, int ns, basic phi) :
		bases::shapes::sphere(nd, ns, traits, phi, p) {}
};
