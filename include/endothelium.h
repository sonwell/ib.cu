#pragma once
#include "bases/types.h"
#include "bases/shapes/periodic_sheet.h"
#include "bases/traits.h"

struct endothelium : bases::shapes::periodic_sheet {
private:
	using matrix = bases::matrix;
	static constexpr bases::traits<endothelium> traits;
public:
	static matrix
	shape(const matrix& params)
	{
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto x = 0.5 * (-1 + t / pi);
			auto z = 2 * (-1 + p / pi);
			auto y = (1 - cos(2 * t + 2 * p)) * (1 - cos(-2 * t + 2 * p)) / 100;

			xdata[0 * rows + tid] = x;
			xdata[1 * rows + tid] = y;
			xdata[2 * rows + tid] = z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <typename basic>
	endothelium(int nd, int ns, basic phi) :
		bases::shapes::periodic_sheet(nd, ns, traits, phi) {}
};
