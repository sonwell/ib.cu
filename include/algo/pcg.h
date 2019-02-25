#pragma once
#include "preconditioner.h"
#include "lwps/vector.h"
#include "lwps/matrix.h"
#include "lwps/blas.h"
#include "util/log.h"

namespace algo {
	namespace krylov {
		lwps::vector
		pcg(const preconditioner& pr, const lwps::matrix& m, const lwps::vector& b, double tol)
		{
			auto&& d = solve(pr, b);
			double delta_new = dot(b, d);
			const double eps = tol * tol;
			lwps::vector r = b;
			lwps::vector x(size(b), lwps::fill::zeros);

			int count = 0;
			while (1) {
				auto q = m * d;
				auto nu = dot(d, q);
				if (nu == 0)
					break;
				auto alpha = delta_new / nu;
				lwps::axpy(alpha, d, x);
				lwps::axpy(-alpha, q, r);
				if (abs(r) <= tol)
					break;
				auto s = solve(pr, r);
				auto delta_old = delta_new;
				delta_new = dot(r, s);
				auto beta = delta_new / delta_old;
				lwps::axpy(beta, d, s);
				lwps::swap(d, s);
				++count;
			}
			util::logging::info("pcg iterations: ", count);
			return std::move(x);
		}

		inline lwps::vector
		cg(const lwps::matrix& m, const lwps::vector& b, double tol)
		{
			static typename algo::preconditioner::identity id;
			return pcg(id, m, b, tol);
		}
	}
}
