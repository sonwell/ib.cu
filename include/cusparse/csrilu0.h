#pragma once
#include "types.h"
#include "handle.h"
#include "matrix_descr.h"
#include "csrilu02_info.h"
#include "operation.h"
#include "solve_policy.h"
#include "solve_analysis_info.h"

namespace cusparse {

inline void
csrilu0(handle& h, operation_adaptor trans, int m,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseScsrilu0(h, trans, m, descr_a, values_a,
		starts_a, indices_a, info));
}

inline void
csrilu0(handle& h, operation_adaptor trans, int m,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, solve_analysis_info info)
{
	throw_if_error(cusparseDcsrilu0(h, trans, m, descr_a, values_a,
		starts_a, indices_a, info));
}

inline void
csrilu02_numeric_boost(handle& h, csrilu02_info info, int enable_boost,
		double* tol, float* boost_val)
{
	throw_if_error(cusparseScsrilu02_numericBoost(h, info, enable_boost,
		tol, boost_val));
}

inline void
csrilu02_numeric_boost(handle& h, csrilu02_info info, int enable_boost,
		double* tol, double* boost_val)
{
	throw_if_error(cusparseDcsrilu02_numericBoost(h, info, enable_boost,
		tol, boost_val));
}

inline void
csrilu02_zero_pivot(handle& h, csrilu02_info info, int* position)
{
	throw_if_error(cusparseXcsrilu02_zeroPivot(h, info, position));
}

inline void
csrilu02_buffer_size(handle& h, int m, int nnz,
		const matrix_description descr_a, float* values_a, const int* starts_a,
		const int* indices_a, csrilu02_info info, int* buffer_size)
{
	throw_if_error(cusparseScsrilu02_bufferSize(h, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrilu02_buffer_size(handle& h, int m, int nnz,
		const matrix_description descr_a, double* values_a, const int* starts_a,
		const int* indices_a, csrilu02_info info, int* buffer_size)
{
	throw_if_error(cusparseDcsrilu02_bufferSize(h, m, nnz, descr_a,
		values_a, starts_a, indices_a, info, buffer_size));
}

inline void
csrilu02_buffer_size_ext(handle& h, int m, int nnz,
		const matrix_description descr_a, float* values, const int* starts,
		const int* indices, csrilu02_info info, size_t* buffer_size)
{
	throw_if_error(cusparseScsrilu02_bufferSizeExt(h, m, nnz, descr_a,
		values, starts, indices, info, buffer_size));
}

inline void
csrilu02_buffer_size_ext(handle& h, int m, int nnz,
		const matrix_description descr_a, double* values, const int* starts,
		const int* indices, csrilu02_info info, size_t* buffer_size)
{
	throw_if_error(cusparseDcsrilu02_bufferSizeExt(h, m, nnz, descr_a,
		values, starts, indices, info, buffer_size));
}

inline void
csrilu02_analysis(handle& h, int m, int nnz, const matrix_description descr_a,
		const float* values_a, const int* starts_a, const int* indices_a,
		csrilu02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsrilu02_analysis(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, policy, buffer));
}

inline void
csrilu02_analysis(handle& h, int m, int nnz, const matrix_description descr_a,
		const double* values_a, const int* starts_a, const int* indices_a,
		csrilu02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDcsrilu02_analysis(h, m, nnz, descr_a, values_a,
		starts_a, indices_a, info, policy, buffer));
}

inline void
csrilu02(handle& h, int m, int nnz, const matrix_description descr_a,
		float* values_a, const int* starts_a, const int* indices_a,
		csrilu02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseScsrilu02(h, m, nnz, descr_a, values_a, starts_a,
		indices_a, info, policy, buffer));
}

inline void
csrilu02(handle& h, int m, int nnz, const matrix_description descr_a,
		double* values_a, const int* starts_a, const int* indices_a,
		csrilu02_info info, solve_policy_adaptor policy, void* buffer)
{
	throw_if_error(cusparseDcsrilu02(h, m, nnz, descr_a, values_a, starts_a,
		indices_a, info, policy, buffer));
}

} // namespace cusparse
