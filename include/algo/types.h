#pragma once
#include "linalg/linalg.h"
/*
#include "lwps/vector.h"
#include "lwps/matrix.h"
#include "lwps/kron.h"
#include "lwps/fill.h"
//*/

namespace algo {
/*
using vector = lwps::vector;
using matrix = lwps::matrix;

constexpr inline lwps::fill::constant
fill(double v) { return lwps::fill::constant{v}; }

struct size : lwps::matrix_size {
	using lwps::matrix_size::matrix_size;

	size(const lwps::matrix_base& b) :
		lwps::matrix_size(lwps::size(b)) {}
	size(int rows, int cols) :
		lwps::matrix_size{rows, cols} {}
};/*/

using vector = linalg::vector<linalg::dense<double>>;
using matrix = linalg::matrix<linalg::sparse<double>>;
using sparse_matrix = linalg::matrix<linalg::sparse<double>>;
using dense_matrix = linalg::matrix<linalg::dense<double>>;

using linalg::size;
using linalg::fill;
//*/

static constexpr int indexing_base = 0;

}
