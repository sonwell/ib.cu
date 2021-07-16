#pragma once
#include "algo/symmilu.h"
#include "algo/redblack.h"
#include "solvers/types.h"
#include "base.h"

namespace solvers {
namespace preconditioners {

struct ilu : algo::symmilu, preconditioner {
	virtual dense::vector
	operator()(dense::vector b) const
	{
		return algo::symmilu::operator()(std::move(b));
	}

	template <typename grid_type>
	ilu(const grid_type& grid, const sparse::matrix& m) :
		algo::symmilu{m, new algo::redblack{grid}} {}
};

} // namespace preconditioners
} // namespace solvers
