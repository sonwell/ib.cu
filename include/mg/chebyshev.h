#pragma once
#include <utility>
#include "util/launch.h"
#include "algo/chebyshev.h"
#include "types.h"
#include "smoother.h"

namespace mg {

class chebyshev : public algo::chebyshev, public smoother {
private:
	template <typename domain_type>
	std::pair<double, double>
	range(const domain_type&, const matrix& m)
	{
		static constexpr auto correction = 1 << domain_type::ndim;
		auto [a, b] = algo::gershgorin(m);
		if (abs(a) > abs(b))
			std::swap(a, b);
		return {a + (b - a) / correction, b};
	}

	chebyshev(std::pair<double, double> range, const matrix& m) :
		algo::chebyshev(std::get<0>(range), std::get<1>(range), m) {}
protected:
	using algo::chebyshev::op;
public:
	virtual vector
	operator()(const vector& b) const
	{
		return algo::chebyshev::operator()(b);
	}

	template <typename domain_type>
	chebyshev(const domain_type& domain, const matrix& m) :
		chebyshev(range(domain, m), m) {}
};

} // namespace mg
