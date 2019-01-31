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
			static constexpr auto dimensions = domain_type::ndim;
			auto range = algo::gershgorin(m);
			auto a = std::get<0>(range);
			auto b = std::get<1>(range);
			if (abs(a) > abs(b))
				std::swap(a, b);
			return {a + (b - a) / (1 << dimensions), b};
		}

		chebyshev(std::pair<double, double> range, const lwps::matrix& m) :
			algo::chebyshev(std::get<0>(range), std::get<1>(range), m) {}
	protected:
		using algo::chebyshev::op;
	public:
		/*virtual lwps::vector
		operator()(const lwps::vector& x, const lwps::vector& b) const
		{
			auto&& r = b - op * x;
			return algo::chebyshev::operator()(r) + x;
		}*/

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
