#pragma once
#include "types.h"
#include <cmath>

namespace lwps {
	class vector;
	class matrix;

	void swap(vector&, vector&);
	void swap(matrix&, matrix&);
	void copy(const vector&, vector&);
	void copy(const matrix&, matrix&);
	void scal(value_type, vector&);
	void scal(value_type, matrix&);
	void axpy(value_type, const vector&, vector&);
	void axpy(value_type, const matrix&, matrix&);
	double dot(const vector&, const vector&);
	inline double nrm2(const vector& v) { return sqrt(dot(v, v)); }
	inline double abs(const vector& v) { return nrm2(v); }

	void gemv(value_type, const matrix&, const vector&, value_type, vector&);
}
