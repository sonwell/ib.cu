#pragma once
#include <cmath>
#include "util/log.h"
#include "linalg/fill.h"
#include "preconditioners/base.h"

#include "types.h"

namespace solvers {

inline dense::vector
pcg(const preconditioner& pr, const sparse::matrix& m, dense::vector r, double tol)
{
	auto n = r.rows();
	auto d = solve(pr, r);
	double delta_new = dot(r, d);
	dense::vector x{n, linalg::zero};

	int count = 0;
	dense::vector q{n};
	while (abs(d) > tol) {
		gemv(1.0, m, d, 0.0, q);
		auto nu = dot(d, q);
		if (std::abs(nu) < tol * tol) {
			// Typically means we have improved the approximation by a factor of
			// 1/machine epsilon but still have not converged.
			util::logging::warn("pcg bailing out early:〈P⁻¹r,AP⁻¹r〉= ", nu);
			break;
		}
		auto alpha = delta_new / nu;
		axpy(alpha, d, x);
		axpy(-alpha, q, r);

		q = solve(pr, r);
		auto delta_old = delta_new;
		delta_new = dot(r, q);
		auto beta = delta_new / delta_old;
		axpy(beta, d, q);
		swap(d, q);
		++count;
	}
	util::logging::debug("pcg iterations: ", count);
	return x;
}

inline dense::vector
cg(const sparse::matrix& m, dense::vector b, double tol)
{
	static typename preconditioners::identity id;
	return pcg(id, m, std::move(b), tol);
}

} // namespace solvers
