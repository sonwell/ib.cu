#pragma once
#include <cmath>
#include <utility>
#include "util/launch.h"
#include "lwps/vector.h"
#include "lwps/matrix.h"
#include "algo/chebyshev.h"
#include "smoother.h"

namespace mg {
	class chebyshev : public algo::chebyshev, public smoother {
	private:
		template <typename domain_type>
		double
		eigenvalue(const domain_type& domain)
		{
			return -4.0 * domain_type::ndim * pow(domain.resolution(), 2);
		}
	public:
		virtual lwps::vector
		operator()(const lwps::vector& x, const lwps::vector& b) const
		{
			auto&& r = b - op * x;
			return algo::chebyshev::operator(r) + x;
		}

		virtual lwps::vector
		operator()(const lwps::vector& b) const
		{
			return algo::chebyshev::operator(b);
		}

		template <typename domain_type>
		chebyshev(const domain_type& domain, const lwps::matrix& m) :
			algo::chebyshev(eigenvalue(domain) / (1 << domain_type::ndim), eigenvalue(domain), m) {}
	};
}
