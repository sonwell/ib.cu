#pragma once
#include "preconditioner.h"
#include "lwps/vector.h"
#include "lwps/matrix.h"

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
			while (std::abs(delta_new) > eps) {
				auto q = m * d;
				auto nu = dot(d, q);
				if (nu == 0)
					break;
				auto alpha = delta_new / nu;
				x += alpha * d;
				r -= alpha * std::move(q);
				auto s = solve(pr, r);
				auto delta_old = delta_new;
				delta_new = dot(r, s);
				auto beta = delta_new / delta_old;
				d = beta * std::move(d) + s;
				++count;
			}
			std::cout << count << std::endl;
			return std::move(x);
		}

		inline lwps::vector
		cg(const lwps::matrix& m, const lwps::vector& b, double tol)
		{
			static constexpr typename algo::preconditioner::identity id;
			return pcg(id, m, b, tol);
		}
	}
}
