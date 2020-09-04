#pragma once
#include "torus.h"
#include "bases/traits.h"
#include "bases/polynomials.h"

namespace bases {
namespace shapes {

using namespace bases::meta;

// The periodic sheet is like a torus, but it is unrolled flat, so uses up to
// linear polynomials to compute its geometry.
struct periodic_sheet : torus {
private:
	static constexpr bases::traits<periodic_sheet> traits;
	static constexpr bases::polynomials<1> p;
protected:
	using torus::shape;
	template <meta::traits traits, meta::basic interp, meta::basic eval>
	periodic_sheet(int n, int m, traits tr, interp phi, eval psi) :
		torus(n, m, tr, phi, psi, p) {}

	template <meta::traits traits, meta::basic basic>
	periodic_sheet(int n, int m, traits tr, basic phi) :
		periodic_sheet(n, m, tr, phi, phi, p) {}
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
			auto x = t / (2 * pi);
			auto y = p / (2 * pi);
			auto z = 0.0;

			xdata[0 * rows + tid] = x;
			xdata[1 * rows + tid] = y;
			xdata[2 * rows + tid] = z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	template <meta::basic interp, meta::basic eval>
	periodic_sheet(int n, int m, interp phi, eval psi) :
		periodic_sheet(n, m, traits, phi, psi) {}

	template <meta::basic basic>
	periodic_sheet(int n, int m, basic phi) :
		periodic_sheet(n, m, traits, phi, phi) {}
};

} // namespace shapes
}
