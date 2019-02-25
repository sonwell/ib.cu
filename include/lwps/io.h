#pragma once
#include <ostream>
#include <iomanip>
#include "types.h"

namespace lwps {
namespace io {
namespace style {
	struct base {
		std::string begin_vector;
		std::string vector_entry_delimiter;
		std::string end_vector;

		std::string begin_matrix;
		std::string matrix_entry_delimiter;
		std::string end_matrix;

		std::string begin_matrix_row;
		std::string matrix_row_delimiter;
		std::string end_matrix_row;

		virtual void prepare(std::ostream& out) const {}

		virtual void
		value(std::ostream& out, lwps::value_type v)
		{
			out << std::setw(7) << v;
		}


		base(const std::string& bv, const std::string& ved, const std::string& ev,
				const std::string& bm, const std::string& med, const std::string& em,
				const std::string& bmr, const std::string& mrd, const std::string& emr) :
			begin_vector(bv), vector_entry_delimiter(ved), end_vector(ev),
			begin_matrix(bm), matrix_entry_delimiter(med), end_matrix(em),
			begin_matrix_row(bmr), matrix_row_delimiter(mrd), end_matrix_row(emr) {}
	};

	struct none : base {
		none() : base{
			"", ", ", "",
			"", ", ", "",
			"", "\n", ""
		} {}
	};

	struct matlab : base {
		matlab() : base{
			"[", ", ", "]'",
			"[", ", ", "]",
			" ", ";\n", ""
		} {}
	};

	struct python : base {
		python() : base{
			"[", ", ", "]",
			"[", ", ", "]",
			" [", ",\n", "]"
		} {}
	};

	struct numpy : base {
		numpy() : base{
			"array([", ", ", "])",
			"array([", ", ", "])",
			"       [", ",\n", "]"
		} {}
	};
}

	inline style::base*&
	get_internal_style()
	{
		static style::none default_style;
		static style::base* internal = &default_style;
		return internal;
	}

	inline style::base*
	get_style()
	{
		return get_internal_style();
	}

	template <typename style_type>
	void
	set_style(style_type&& style)
	{
		static style_type init{style};
		get_internal_style() = &init;
	}
}
}
