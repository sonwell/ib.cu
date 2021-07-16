#pragma once
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "linalg/dense.h"
#include "linalg/sparse.h"

namespace solvers {

namespace dense {
using matrix = linalg::matrix<linalg::dense<double>>;
using vector = linalg::vector<linalg::dense<double>>;
} // namespace dense

namespace sparse {
using matrix = linalg::matrix<linalg::sparse<double>>;
using vector = linalg::vector<linalg::sparse<double>>;
} // namespace sparse

static constexpr auto indexing_base = 0;

} // namespace solvers
