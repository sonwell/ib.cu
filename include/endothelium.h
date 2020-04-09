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
		constexpr double height = 2_um;
		constexpr double stretch = 16_um;
		auto k = [=] __device__ (auto x) -> point
		{
			auto u = x[0] / (2 * pi);
			auto v = x[1] / (2 * pi);
			auto r = 2 * (u - 0.5 * v);
			auto s = 2 * (u + 0.5 * v);
			auto w = 0.25 * (1 + cos(4 * pi * r)) * (1 + cos(4 * pi * s));
			return {stretch * v, height * (w + 0.5), stretch * u};
		};
		return base::shape(params, k);
	}

	template <typename interp, typename eval,
			 typename = std::enable_if_t<bases::is_basic_function_v<interp>>,
			 typename = std::enable_if_t<bases::is_basic_function_v<eval>>>
	endothelium(int nd, int ns, interp phi, eval psi) :
		bases::shapes::periodic_sheet(nd, ns, traits, phi, psi) {}

	template <typename basic,
			 typename = std::enable_if_t<bases::is_basic_function_v<basic>>>
	endothelium(int nd, int ns, basic phi) :
		endothelium(nd, ns, phi, phi) {}
};
