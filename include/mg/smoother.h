#pragma once
#include "types.h"

namespace mg {

struct smoother {
	virtual vector operator()(vector) const = 0;
	virtual ~smoother() = default;
};

inline auto solve(const smoother& sm, vector b) { return sm(std::move(b)); }

} // namespace mg
