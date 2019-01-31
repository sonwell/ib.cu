#pragma once
#include <type_traits>
#include "util/getset.h"
#include "types.h"
#include "exceptions.h"

namespace cusparse {
	enum class diagonal_type : std::underlying_type_t<diagonal_type_t> {
		unit = CUSPARSE_DIAG_TYPE_UNIT,
		non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT
	};

	enum class fill_mode : std::underlying_type_t<fill_mode_t> {
		lower = CUSPARSE_FILL_MODE_LOWER,
		upper = CUSPARSE_FILL_MODE_UPPER
	};

	enum class index_base : std::underlying_type_t<index_base_t> {
		zero = CUSPARSE_INDEX_BASE_ZERO,
		one = CUSPARSE_INDEX_BASE_ONE
	};

	enum class matrix_type : std::underlying_type_t<matrix_type_t> {
		general = CUSPARSE_MATRIX_TYPE_GENERAL,
		symmetric = CUSPARSE_MATRIX_TYPE_SYMMETRIC,
		hermitian = CUSPARSE_MATRIX_TYPE_HERMITIAN,
		triangular = CUSPARSE_MATRIX_TYPE_TRIANGULAR
	};

	class matrix_description : public type_wrapper<mat_descr_t> {
	private:
		using dt_t = cusparse::diagonal_type;
		using fm_t = cusparse::fill_mode;
		using ib_t = cusparse::index_base;
		using mt_t = cusparse::matrix_type;
	protected:
		using type_wrapper<mat_descr_t>::data;

		dt_t get_dt() const { return static_cast<dt_t>(cusparseGetMatDiagType(data)); }
		void set_dt(const dt_t& v) { throw_if_error(cusparseSetMatDiagType(data, (diagonal_type_t) v)); }
		fm_t get_fm() const { return static_cast<fm_t>(cusparseGetMatFillMode(data)); }
		void set_fm(const fm_t& v) { throw_if_error(cusparseSetMatFillMode(data, (fill_mode_t) v)); }
		ib_t get_ib() const { return static_cast<ib_t>(cusparseGetMatIndexBase(data)); }
		void set_ib(const ib_t& v) { throw_if_error(cusparseSetMatIndexBase(data, (index_base_t) v)); }
		mt_t get_mt() const { return static_cast<mt_t>(cusparseGetMatType(data)); }
		void set_mt(const mt_t& v) { throw_if_error(cusparseSetMatType(data, (matrix_type_t) v)); }
	public:
		util::getset<dt_t> diagonal_type;
		util::getset<fm_t> fill_mode;
		util::getset<ib_t> index_base;
		util::getset<mt_t> matrix_type;

		matrix_description() :
			diagonal_type([&] () { return get_dt(); }, [&] (const dt_t& v) { set_dt(v); }),
			fill_mode    ([&] () { return get_fm(); }, [&] (const fm_t& v) { set_fm(v); }),
			index_base   ([&] () { return get_ib(); }, [&] (const ib_t& v) { set_ib(v); }),
			matrix_type  ([&] () { return get_mt(); }, [&] (const mt_t& v) { set_mt(v); })
		{ throw_if_error(cusparseCreateMatDescr(&data)); }
		matrix_description(const matrix_description& o) :
			matrix_description()
		{ throw_if_error(cusparseCopyMatDescr(data, o.data)); }
		~matrix_description() { throw_if_error(cusparseDestroyMatDescr(data)); }
	};
}
