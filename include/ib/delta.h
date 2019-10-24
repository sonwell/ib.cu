#pragma once
#include "util/math.h"

namespace ib {

struct cosine_delta {
private:
	static constexpr auto pi2 = M_PI_2;
public:
	constexpr auto
	operator()(double r) const
	{
		using util::math::cos;
		return 0.25 * (1 + cos(pi2 * r));
	}
};

}
