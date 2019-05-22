#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "csric02_info.h"
#include "operation.h"
#include "solve_policy.h"
#include "solve_analysis_info.h"

namespace cusparse {

inline void
csric0(handle& h, operation_adaptor trans, int m,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseScsric0(h, trans, m, descr_a, values_a, starts_a,
		indices_a, info));
}

inline void
csric0(handle& h, operation_adaptor trans, int m,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseDcsric0(h, trans, m, descr_a, values_a, starts_a,
		indices_a, info));
}

inline void
csric02_zero_pivot(handle& h, csric02_info info, int* position)
{
	throw_if_error(cusparseXcsric02_zeroPivot(h, info, position));
}

inline void
csric02_buffer_size(handle& h, int m, int nnz, const matrix_description descr_a,
		float* values_a, const int* starts_a, const int* indices_a,
		csric02_info info, int* buffer_size)
{
	throw_if_error(cusparseScsric02_bufferSize(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, buffer_size));
}

inline void
csric02_buffer_size(handle& h, int m, int nnz, const matrix_description descr_a,
		double* values_a, const int* starts_a, const int* indices_a,
		csric02_info info, int* buffer_size)
{
	throw_if_error(cusparseDcsric02_bufferSize(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, buffer_size));
}

inline void
csric02_buffer_size_ext(handle& h, int m, int nnz,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, csric02_info info, size_t* buffer_size)
{
	throw_if_error(cusparseScsric02_bufferSizeExt(h, m, nnz, descr_a,
		values, starts, indices, info, buffer_size));
}

inline void
csric02_buffer_size_ext(handle& h, int m, int nnz,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, csric02_info info, size_t* buffer_size)
{
	throw_if_error(cusparseDcsric02_bufferSizeExt(h, m, nnz, descr_a,
		values, starts, indices, info, buffer_size));
}

inline void
csric02_analysis(handle& h, int m, int nnz, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		csric02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsric02_analysis(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, policy, buffer));
}

inline void
csric02_analysis(handle& h, int m, int nnz, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		csric02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDcsric02_analysis(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, policy, buffer));
}

inline void
csric02(handle& h, int m, int nnz, const matrix_description descr_a,
		float* values_a, to be the preconditioner M valuesconst int* starts_a,
		const int* indices_a, csric02_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseScsric02(h, m, nnz, descr_a, values_a,
		be the preconditioner M valuesconst int* starts_a, indices_a, info,
		policy, buffer));
}

inline void
csric02(handle& h, int m, int nnz, const matrix_description descr_a,
		double* values_a, to be the preconditioner M valuesconst int* starts_a,
		const int* indices_a, csric02_info info, solve_policy_adaptor policy,
		void* buffer)
{
	throw_if_error(cusparseDcsric02(h, m, nnz, descr_a, values_a,
		be the preconditioner M valuesconst int* starts_a, indices_a, info,
		policy, buffer));
}

} // namespace cusparse
