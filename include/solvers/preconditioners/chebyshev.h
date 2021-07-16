#pragma once
#include "algo/chebyshev.h"
#include "solvers/types.h"
#include "base.h"

namespace solvers {
namespace preconditioners {

struct chebyshev : algo::chebyshev, preconditioner {
private:
	chebyshev(std::pair<double, double> range, algo::matrix m) :
		algo::chebyshev(range.second, range.first, std::move(m)) {}
protected:
	using algo::chebyshev::m;
public:
	const sparse::matrix& op() const { return m; }

	virtual dense::vector
	operator()(dense::vector b) const
	{
		return algo::chebyshev::operator()(std::move(b));
	}

	chebyshev(sparse::matrix m) :
		chebyshev(algo::gershgorin(m), std::move(m)) {}
};

} // namespace preconditioners
} // namespace solvers
