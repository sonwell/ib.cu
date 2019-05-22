#pragma once
#include "types.h"

namespace linalg {

template <typename> class sparse;
template <typename> class dense;
template <typename> class vector;
template <typename> class matrix;

template <template <typename> class container,
		 template <typename> class layout, typename vtype>
void swap(container<layout<vtype>>&, container<layout<vtype>>&);

template <template <typename> class container,
		 template <typename> class layout, typename ftype, typename ttype>
void copy(const container<layout<ftype>>&, container<layout<ttype>>&);

template <template <typename> class container,
		 template <typename> class layout, typename vtype>
void scal(scalar<vtype>, container<layout<vtype>>&);

template <template <typename> class container, typename vtype>
void axpy(scalar<vtype>, const container<dense<vtype>>&, container<dense<vtype>>&);

template <typename vtype>
void axpy(scalar<vtype>, const matrix<sparse<vtype>>&, matrix<sparse<vtype>>&);

template <typename vtype>
void axpy(scalar<vtype>, const vector<sparse<vtype>>&, vector<dense<vtype>>&);

template <template <typename> class layout, typename vtype>
vtype dot(const vector<layout<vtype>>&, const vector<layout<vtype>>&);

template <typename vtype>
vtype dot(const vector<sparse<vtype>>&, const vector<dense<vtype>>&);

template <typename vtype>
inline vtype
dot(const vector<dense<vtype>>& left, const vector<sparse<vtype>>& right)
{
	return dot(right, left);
}

template <template <typename> class layout, typename vtype>
vtype nrm2(const vector<layout<vtype>>&);

template <template <typename> class layout, typename vtype>
inline vtype abs(const vector<layout<vtype>>& v) { return nrm2(v); }

template <typename vtype>
void gemv(scalar<vtype>, const matrix<dense<vtype>>&, const vector<dense<vtype>>&,
		scalar<vtype>, vector<dense<vtype>>&);

template <typename vtype>
void gemv(scalar<vtype>, const matrix<sparse<vtype>>&, const vector<dense<vtype>>&,
		scalar<vtype>, vector<dense<vtype>>&);

template <typename vtype>
void gemv(scalar<vtype>, const matrix<sparse<vtype>>&, const vector<sparse<vtype>>&,
		scalar<vtype>, vector<dense<vtype>>&);

template <typename vtype>
void gemm(scalar<vtype>, const matrix<dense<vtype>>&, const matrix<dense<vtype>>&,
		scalar<vtype>, matrix<dense<vtype>>&);

template <typename vtype>
void gemm(scalar<vtype>, const matrix<sparse<vtype>>&, const matrix<dense<vtype>>&,
		scalar<vtype>, matrix<dense<vtype>>&);

template <typename vtype>
void gemm(scalar<vtype>, const matrix<sparse<vtype>>&, const matrix<sparse<vtype>>&,
		scalar<vtype>, matrix<sparse<vtype>>&);

template <typename vtype>
void hadamard(const vector<dense<vtype>>&, const vector<dense<vtype>>&);

template <typename vtype>
void hadamard(const vector<sparse<vtype>>&, const vector<sparse<vtype>>&);

}
