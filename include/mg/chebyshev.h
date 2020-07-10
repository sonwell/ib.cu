#pragma once
#include <utility>
#include "util/launch.h"
#include "algo/chebyshev.h"
#include "types.h"
#include "smoother.h"

namespace mg {

// Chebyshev iteration as a MG smoother
class chebyshev : public algo::chebyshev, public smoother {
private:
	template <typename grid_type>
	std::pair<double, double>
	range(const grid_type&, const matrix& m)
	{
		static constexpr auto correction = 1 << grid_type::dimensions;
		auto [a, b] = algo::gershgorin(m);
		if (abs(a) > abs(b))
			std::swap(a, b);
		return {a + (b - a) / correction, b};
	}

	chebyshev(std::pair<double, double> range, const matrix& m) :
		algo::chebyshev(std::get<0>(range), std::get<1>(range), m) {}
public:
	virtual vector
	operator()(vector b) const
	{
		return algo::chebyshev::operator()(std::move(b));
	}

	template <typename grid_type>
	chebyshev(const grid_type& grid, const matrix& m) :
		chebyshev(range(grid, m), m) {}
};

} // namespace mg
