#pragma once
#include <stdexcept>
#include <utility>
#include "util/math.h"
#include "util/wrapper.h"
#include "cell.h"
#include "domain.h"
#include "boundary.h"
#include "dimension.h"
#include "exceptions.h"
#include "correction.h"
#include "types.h"

namespace fd {
namespace __1 {

typedef struct { bool row_end, col_end; double value; } entry;

inline matrix
single_entry(int rows, int cols, double scale, entry e)
{
	enum { nt = 128, vt = 7 };
	auto value = scale * e.value;
	if (!rows || !cols || !value)
		return matrix{rows, cols};

	auto row = (rows && e.row_end) ? rows-1 : 0;
	auto col = (cols && e.col_end) ? cols-1 : 0;
	matrix result{rows, cols, 1};

	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();

	auto k = [=] __device__ (int tid, int row, int col,
			int* starts, int* indices, double* values)
	{
		starts[tid] = tid > row;
		if (!tid) {
			indices[0] = col + indexing_base;
			values[0] = value;
		}
	};
	util::transform<nt, vt>(k, rows+1, row, col, starts, indices, values);
	return result;
}

struct coefficients {
	using params = std::array<double, 2>;

	// Consider a boundary condition ɑu + β∂u = ɣ. Let
	// u₋₁ = Aɣ + Bu₀ be an approximation of the value at
	// ghost cell u₋₁.

	// Coefficient of the boundary condition RHS (ɣ) in the approximation
	// at the ghost point (i.e., A).
	double boundary;

	// Coefficient of the interior point (u₀) in the approximation at the
	// ghost point (i.e., B).
	double interior;

	// Coefficient of 0th-order error in approximating the 3-pt
	// Laplacian at the interior point using linear approximations
	// to u and ∂u at the boundary. (There is no 1st-order error
	// coefficient due to symmetry.)
	double laplacian;

	template <typename boundary_type>
	constexpr coefficients(const boundary_type& boundary, double h, double s) :
		coefficients(boundary.params(), h, 1 - util::math::modulo(1 - s, 1)) {}
	constexpr coefficients(const boundary::periodic& boundary, double, double) :
		boundary(0), interior(0), laplacian(0) {}
private:
	constexpr coefficients(const params& p, double h, double s) :
		coefficients(p, h, s, p[1] - p[0] * s * h) {}
	constexpr coefficients(const params& p, double h, double s, double denom) :
		boundary(-h / denom), interior((p[1] + p[0] * h * (1 - s)) / denom),
		laplacian((-(p[1] * (1 - 2 * s) - p[0] * s * h * (1 - s))) / (2 * denom)) {}
};


typedef struct { double lower, upper; } params;

} // namespace __1

template <typename> struct discretization;

template <typename lbt, typename ubt>
struct discretization<fd::dimension<lbt, ubt>> : fd::dimension<lbt, ubt> {
private:
	using entry = __1::entry;
	using params = __1::params;
	using base = fd::dimension<lbt, ubt>;

	template <bool is_lower>
	constexpr entry
	interior(boundary::tag<is_lower>) const
	{
		if constexpr (!solid_boundary) return {!is_lower, is_lower, 1.0};
		else return {!is_lower, !is_lower, weights[!is_lower].interior};
	}

	template <bool is_lower>
	constexpr entry
	boundary(boundary::tag<is_lower>) const
	{
		if constexpr (!solid_boundary) return {!is_lower, is_lower, 0.0};
		else return {!is_lower, !is_lower, weights[!is_lower].boundary};
	}
public:
	using base::solid_boundary;
	constexpr auto alignment() const { return _alignment; }
	constexpr auto on_boundary() const { return _alignment.on_boundary(); }
	constexpr auto shift() const { return _alignment.shift + (solid_boundary && on_boundary()); }
	constexpr auto cells() const { return _cells; }
	constexpr auto points() const { return _points; }
	constexpr auto resolution() const { return _resolution; }
	constexpr params boundary() const { return {weights[0].boundary, weights[1].boundary}; }
	constexpr params interior() const { return {weights[0].interior, weights[1].interior}; }

	template <std::size_t n = 0>
	constexpr params
	coefficient(correction::order<n> = {}) const
	{
		static_assert(n < 3, "error coefficients only computed up to second order");
		if constexpr (n < 2) return {0., 0.};
		else return {weights[0].laplacian, weights[1].laplacian};
	}

	template <bool lower>
	auto
	interior(int rows, int cols, double scale, boundary::tag<lower> tag) const
	{
		return single_entry(rows, cols, scale, interior(tag));
	}

	template <bool lower>
	auto
	boundary(int rows, int cols, double scale, boundary::tag<lower> tag) const
	{
		return single_entry(rows, cols, scale, boundary(tag));
	}

	constexpr double
	units(double x) const
	{
		auto s = shift();
		auto r = resolution();
		return base::clamp(x - s / r) * r + s;
	}

	constexpr int
	index(int i) const
	{
		if constexpr (solid_boundary) return i;
		else return (i + _cells) % _cells;
	}

	constexpr discretization(const base& dimension,
			struct alignment al, double resolution) :
		base(dimension), _alignment(al), _resolution(resolution),
		_cells(dimension.length() * resolution + 0.5),
		_points(_cells - (al.on_boundary() && solid_boundary)),
		weights{{dimension.lower(), 1/resolution, al.shift},
		        {dimension.upper(), -1/resolution, 1-al.shift}} {}

	constexpr discretization(const discretization& disc,
			double resolution) :
		discretization(disc, disc.alignment(), resolution) {}

	constexpr discretization(const discretization& disc,
			struct alignment al) :
		discretization(disc, al, disc.resolution()) {}
private:
	struct alignment _alignment;
	double _resolution;
	int _cells;
	int _points;
	__1::coefficients weights[2];
};

template <typename dimension_type>
discretization(const dimension_type&, alignment, double)
	-> discretization<dimension_type>;

template <typename dimension_type>
discretization(const discretization<dimension_type>&, alignment)
	-> discretization<dimension_type>;

template <typename dimension_type>
discretization(const discretization<dimension_type>&, double)
	-> discretization<dimension_type>;

} // namespace ib
