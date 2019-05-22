#pragma once
#include "types.h"

namespace algo {

struct preconditioner {
	struct identity;
	virtual vector operator()(const vector&) const = 0;
	virtual ~preconditioner() {}
};

struct preconditioner::identity : preconditioner {
	virtual vector
	operator()(const vector& v) const
	{
		return v;
	}
};

inline auto
solve(const preconditioner& pr, const vector& v)
{
	return pr(v);
}

} // namespace algo
