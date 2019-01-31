#pragma once
#include "lwps/vector.h"

namespace algo {
	struct preconditioner {
		struct identity;
		virtual lwps::vector operator()(const lwps::vector&) const = 0;
	};

	struct preconditioner::identity : preconditioner {
		virtual lwps::vector
		operator()(const lwps::vector& v) const
		{
			return v;
		}
	};

	inline auto
	solve(const preconditioner& pr, const lwps::vector& v)
	{
		return pr(v);
	}
}

