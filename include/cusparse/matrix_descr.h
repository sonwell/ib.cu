#pragma once
#include <ostream>
#include "util/getset.h"
#include "util/adaptor.h"
#include "types.h"
#include "exceptions.h"
#include "diagonal_type.h"
#include "fill_mode.h"
#include "index_base.h"
#include "matrix_type.h"

namespace cusparse {

inline void
create(mat_descr_t& descr)
{
	throw_if_error(cusparseCreateMatDescr(&descr));
}

inline void
destroy(mat_descr_t& descr)
{
	throw_if_error(cusparseDestroyMatDescr(descr));
}

inline void
copy(mat_descr_t& dest, const mat_descr_t& src)
{
	throw_if_error(cusparseCopyMatDescr(dest, src));
}

class matrix_description : public cusparse::type_wrapper<mat_descr_t> {
private:
	using base = cusparse::type_wrapper<mat_descr_t>;
	using dt_a = diagonal_adaptor;
	using fm_a = fill_mode_adaptor;
	using ib_a = index_base_adaptor;
	using mt_a = matrix_type_adaptor;
protected:
	using base::value;

	dt_a dt() { return get_diag_type(value); }
	void dt(const dt_a& v) { set_diag_type(value, v); }
	fm_a fm() { return get_fill_mode(value); }
	void fm(const fm_a& v) { set_fill_mode(value, v); }
	ib_a ib() { return get_index_base(value); }
	void ib(const ib_a& v) { set_index_base(value, v); }
	mt_a mt() { return get_matrix_type(value); }
	void mt(const mt_a& v) { set_matrix_type(value, v); }
public:
	util::getset<dt_a> diagonal_type;
	util::getset<fm_a> fill_mode;
	util::getset<ib_a> index_base;
	util::getset<mt_a> matrix_type;

	matrix_description() :
		base(),
		diagonal_type([&] () { return dt(); }, [&] (const dt_a& v) { dt(v); }),
		fill_mode    ([&] () { return fm(); }, [&] (const fm_a& v) { fm(v); }),
		index_base   ([&] () { return ib(); }, [&] (const ib_a& v) { ib(v); }),
		matrix_type  ([&] () { return mt(); }, [&] (const mt_a& v) { mt(v); }) {}
	matrix_description(const mat_descr_t& descr) :
		matrix_description() { cusparse::copy(value, descr); }
	matrix_description(const matrix_description& o) :
		matrix_description(o.value) {}
};

inline std::ostream&
operator<<(std::ostream& out, const matrix_description& descr)
{
	return out << "diagonal: " << descr.diagonal_type << '\n'
	           << "fill mode: " << descr.fill_mode << '\n'
	           << "indexing base: " << descr.index_base << '\n'
	           << "type: " << descr.matrix_type;
}

} // namespace cublas
