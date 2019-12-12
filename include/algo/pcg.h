#pragma once
#include "util/log.h"
#include "types.h"
#include "preconditioner.h"

namespace algo {
namespace krylov {

inline vector
pcg(const preconditioner& pr, const matrix& m, const vector& b, double tol)
{
	auto d = solve(pr, b);
	double delta_new = dot(b, d);
	vector r = b;
	vector x = 0 * b;

	int count = 0;
	while (abs(r) > tol) {
		auto q = m * d;
		auto nu = dot(d, q);
		if (nu == 0)
			break;
		auto alpha = delta_new / nu;
		axpy(alpha, d, x);
		axpy(-alpha, q, r);
		auto s = solve(pr, r);
		auto delta_old = delta_new;
		delta_new = dot(r, s);
		auto beta = delta_new / delta_old;
		axpy(beta, d, s);
		swap(d, s);
		++count;
	}
	util::logging::info("pcg iterations: ", count);
	return x;
}

inline vector
cg(const matrix& m, const vector& b, double tol)
{
	static typename algo::preconditioner::identity id;
	return pcg(id, m, b, tol);
}

} // namespace krylov
} // namespace algo
