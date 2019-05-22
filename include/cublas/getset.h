#pragma once
#include "types.h"
#include "exceptions.h"

namespace cublas {

// vector copies

template <typename from_type, typename to_type>
inline void
get_vector(int n, const from_type* from, int incf, to_type* to, int inct)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"vector types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasGetVector(n, size, from, incf, to, inct));
}

template <typename from_type, typename to_type>
inline void
get_vector(int n, const from_type* from, int incf, to_type* to, int inct,
		stream_t stream)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"vector types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasGetVectorAsync(n, size, from, incf, to, inct, stream));
}

template <typename from_type, typename to_type>
inline void
set_vector(int n, const from_type* from, int incf, to_type* to, int inct)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"vector types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasSetVector(n, size, from, incf, to, inct));
}

template <typename from_type, typename to_type>
inline void
set_vector(int n, const from_type* from, int incf, to_type* to, int inct,
		stream_t stream)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"vector types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasSetVectorAsync(n, size, from, incf, to, inct,
				stream));
}

// matrix copies

template <typename from_type, typename to_type>
inline void
get_matrix(int rows, int cols, const from_type* from, int ldf, to_type* to,
		int ldt)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"matrix types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasGetMatrix(rows, cols, size, from, ldf, to, ldt));
}

template <typename from_type, typename to_type>
inline void
get_matrix(int rows, int cols, const from_type* from, int ldf, to_type* to,
		int ldt, stream_t stream)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"matrix types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasGetMatrixAsync(rows, cols, size, from, ldf, to,
				ldt, stream));
}

template <typename from_type, typename to_type>
inline void
set_matrix(int rows, int cols, const from_type* from, int ldf, to_type* to,
		int ldt)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"matrix types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasSetMatrix(rows, cols, size, from, ldf, to, ldt));
}

template <typename from_type, typename to_type>
inline void
set_matrix(int rows, int cols, const from_type* from, int ldf, to_type* to,
		int ldt, stream_t stream)
{
	static_assert(sizeof(from_type) == sizeof(to_type),
			"matrix types must be of the same size");
	static constexpr auto size = sizeof(from_type);
	throw_if_error(cublasSetMatrixAsync(rows, cols, size, from, ldf, to,
				ldt, stream));
}

} // namespace cublas
