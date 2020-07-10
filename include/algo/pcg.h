#pragma once
#include "util/log.h"
#include "types.h"
#include "preconditioner.h"

namespace algo {
namespace krylov {

inline vector
pcg(const preconditioner& pr, const matrix& m, vector r, double tol)
{
	auto n = r.rows();
	auto d = solve(pr, r);
	double delta_new = dot(r, d);
	vector x{n, linalg::zero};

	int count = 0;
	vector q{n};
	while (abs(d) > tol) {
		gemv(1.0, m, d, 0.0, q);
		auto nu = dot(d, q);
		if (abs(nu) < tol * tol) {
			// Typically means we have improved the approximation by a factor of
			// 1/epsilon but still have not converged.
			util::logging::info("pcg bailing out early:〈P⁻¹r,AP⁻¹r〉= ", nu);
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
	util::logging::info("pcg iterations: ", count);
	return x;
}

inline vector
cg(const matrix& m, vector b, double tol)
{
	static typename algo::preconditioner::identity id;
	return pcg(id, m, std::move(b), tol);
}

} // namespace krylov
} // namespace algo
