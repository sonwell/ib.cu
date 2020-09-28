#pragma once
#include "bases/types.h"
#include "bases/shapes/periodic_sheet.h"
#include "bases/traits.h"
#include "units.h"

struct endothelium : bases::shapes::periodic_sheet {
private:
	using base = bases::shapes::periodic_sheet;
	using matrix = bases::matrix;
	static constexpr bases::traits<endothelium> traits;
public:
	static matrix
	shape(const matrix& params)
	{
		using point = std::array<double, 3>;
		constexpr double offset = 0.75_um;
		constexpr double height = 1_um;
		constexpr double stretch = 16_um;
		auto k = [=] __device__ (auto x) -> point
		{
			auto u = x[0] / (2 * pi);
			auto v = x[1] / (2 * pi);
			auto r = 2 * (0.5 * u - v);
			auto s = 2 * (0.5 * u + v);
			auto w = 0.25 * (1 + cos(4 * pi * r)) * (1 + cos(4 * pi * s));
			return {stretch * v, height * w + offset, stretch * u};
		};
		return base::shape(params, k);
	}

	template <bases::meta::basic interp, bases::meta::basic eval>
	endothelium(int n, interp phi, eval psi) :
		bases::shapes::periodic_sheet(n, traits, phi, psi) {}

	template <bases::meta::basic basic>
	endothelium(int n, basic phi) :
		endothelium(n, phi, phi) {}
};
