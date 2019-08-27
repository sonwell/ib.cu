#pragma once
#include "bases/types.h"
#include "bases/shapes/periodic_sheet.h"
#include "bases/traits.h"
#include "units.h"

struct endothelium : bases::shapes::periodic_sheet {
private:
	using matrix = bases::matrix;
	static constexpr bases::traits<endothelium> traits;
public:
	static matrix
	shape(const matrix& params)
	{
		double height = 2_um;
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto u = pdata[0 * rows + tid] / (2 * pi);
			auto v = pdata[1 * rows + tid] / (2 * pi);
			auto r = 2 * (u - 0.5 * v);
			auto s = 2 * (u + 0.5 * v);
			auto t = 0.25 * (1 + cos(4 * pi * r)) * (1 + cos(4 * pi * s));
			double w = height * t;

			xdata[0 * rows + tid] = u;
			xdata[1 * rows + tid] = v;
			xdata[2 * rows + tid] = w;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <typename basic>
	endothelium(int nd, int ns, basic phi) :
		bases::shapes::periodic_sheet(nd, ns, traits, phi) {}
};
