#pragma once
#include <utility>
#include "util/launch.h"
#include "util/functional.h"
#include "algo/chebyshev.h"
#include "types.h"
#include "smoother.h"

namespace mg {

// Chebyshev iteration as a MG smoother
class chebyshev : public algo::chebyshev, public smoother {
private:
	template <typename grid_type>
	double
	proportion(const grid_type& g)
	{
		using namespace util::functional;
		static constexpr auto min = [] (double l, double r) { return l < r ? l : r; };
		static constexpr auto length = [] (const auto& c) { return (double) c.length(); };
		const auto& components = g.components();
		auto lengths = map(length, components);
		auto smallest = apply(partial(foldl, min), lengths) / 2;
		auto prod = [&] (auto ... lengths) { return ((smallest / lengths) * ... * 1.); };
		return apply(prod, lengths);
	}

	template <typename grid_type>
	std::pair<double, double>
	range(const grid_type& g, const matrix& m)
	{
		auto prop = proportion(g);
		auto [a, b] = algo::gershgorin(m);
		if (abs(a) > abs(b))
			std::swap(a, b);
		return {a + (b - a) * prop, b};
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
