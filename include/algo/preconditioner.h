#pragma once
#include "types.h"

namespace algo {

struct preconditioner {
	struct identity;
	virtual vector operator()(vector) const = 0;
	virtual ~preconditioner() {}
};

struct preconditioner::identity : preconditioner {
	virtual vector operator()(vector v) const { return v; }
};

inline auto
solve(const preconditioner& pr, vector v)
{
	return pr(std::move(v));
}

} // namespace algo
