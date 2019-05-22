#pragma once
#include <ostream>
#include "util/adaptor.h"
#include "types.h"

namespace cusparse {

enum class matrix_type { general, symmetric, hermitian, triangular };
using matrix_type_adaptor = util::adaptor<
	util::enum_container<matrix_type_t,
			CUSPARSE_MATRIX_TYPE_GENERAL,
			CUSPARSE_MATRIX_TYPE_SYMMETRIC,
			CUSPARSE_MATRIX_TYPE_HERMITIAN,
			CUSPARSE_MATRIX_TYPE_TRIANGULAR>,
	util::enum_container<matrix_type,
			matrix_type::general,
			matrix_type::symmetric,
			matrix_type::hermitian,
			matrix_type::triangular>>;

inline std::ostream&
operator<<(std::ostream& out, matrix_type v)
{
	switch (v) {
		case matrix_type::general:
			return out << "general";
		case matrix_type::symmetric:
			return out << "symmetric";
		case matrix_type::hermitian:
			return out << "hermitian";
		case matrix_type::triangular:
			return out << "triangular";
	}
	throw util::bad_enum_value();
}

inline matrix_type
get_matrix_type(mat_descr_t& descr)
{
	auto res = cusparseGetMatType(descr);
	return matrix_type_adaptor(res);
}

inline void
set_matrix_type(mat_descr_t& desc, matrix_type_adaptor diag)
{
	cusparseSetMatType(desc, diag);
}

} // namespace cusparse
