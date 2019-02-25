#pragma once
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
		std::pair<double, double>
		range(const domain_type&, const lwps::matrix& m)
		{
			static constexpr auto correction = 1 << domain_type::ndim;
			auto [a, b] = algo::gershgorin(m);
			if (abs(a) > abs(b))
				std::swap(a, b);
			return {a + (b - a) / correction, b};
		}

		chebyshev(std::pair<double, double> range, const lwps::matrix& m) :
			algo::chebyshev(std::get<0>(range), std::get<1>(range), m) {}
	protected:
		using algo::chebyshev::op;
	public:
		virtual lwps::vector
		operator()(const lwps::vector& b) const
		{
			return algo::chebyshev::operator()(b);
		}

		template <typename domain_type>
		chebyshev(const domain_type& domain, const lwps::matrix& m) :
			chebyshev(range(domain, m), m) {}
	};
}
