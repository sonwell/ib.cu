#pragma once
#include "bases/types.h"
#include "bases/shapes/periodic_sheet.h"
#include "bases/traits.h"
#include "units.h"

enum class endothelial_shape {
	elongated = 1,
	cobblestone = 2,
	flat = 3
};

template <endothelial_shape shape_type>
struct endothelium : bases::shapes::periodic_sheet {
private:
	using base = bases::shapes::periodic_sheet;
	using matrix = bases::matrix;
	using traits = bases::traits<endothelium>;
public:
	static constexpr auto elongated = endothelial_shape::elongated;
	static constexpr auto cobblestone = endothelial_shape::cobblestone;
	static constexpr auto flat = endothelial_shape::flat;

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
			double w0;

			if constexpr (shape_type == elongated)
				w0 = cos(x[0] - x[1]) * sin(x[1]/2);
			else if constexpr (shape_type == cobblestone)
				w0 = cos((x[0] - x[1])/2) * cos((x[0] + x[1])/2);
			else
				w0 = 1./2.;

			auto w = w0 * w0;
			return {stretch * v, height * w + offset, stretch * u};
		};
		return base::shape(params, k);
	}

	template <bases::meta::basic interp, bases::meta::basic eval>
	endothelium(int nd, int ns, interp phi, eval psi) :
		bases::shapes::periodic_sheet(nd, ns, traits{}, phi, psi) {}

	template <bases::meta::basic basic>
	endothelium(int nd, int ns, basic phi) :
		endothelium(nd, ns, phi, phi) {}
};
