#pragma once
#include "linalg/linalg.h"

namespace algo {

using vector = linalg::vector<linalg::dense<double>>;
using matrix = linalg::matrix<linalg::sparse<double>>;
using sparse_matrix = linalg::matrix<linalg::sparse<double>>;
using dense_matrix = linalg::matrix<linalg::dense<double>>;

using linalg::size;
using linalg::fill;

static constexpr int indexing_base = 0;

}
