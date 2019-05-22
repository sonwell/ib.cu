#pragma once
#include "types.h"

namespace mg {

struct smoother {
	virtual vector operator()(const vector&) const = 0;
	virtual ~smoother() = default;
};

inline auto solve(const smoother& sm, const vector& b) { return sm(b); }

} // namespace mg
