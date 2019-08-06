#pragma once
#include "linalg/linalg.h"

namespace ib {

using dense = linalg::dense<double>;
using matrix = linalg::matrix<dense>;
using vector = linalg::vector<dense>;

} // namespace ib
