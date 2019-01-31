#pragma once
#include "lwps/vector.h"

namespace mg {
	struct smoother {
		virtual lwps::vector
			operator()(const lwps::vector&, const lwps::vector&) const = 0;
		virtual lwps::vector
			operator()(const lwps::vector&) const = 0;
		virtual ~smoother() = default;
	};

	inline auto
	solve(const smoother& sm, const lwps::vector& b)
	{
		return sm(b);
	}
}
