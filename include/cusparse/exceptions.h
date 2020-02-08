#pragma once
#include <stdexcept>
#include <string>
#include <cusparse.h>
#include <system_error>
#include "types.h"

template <> struct std::is_error_code_enum<cusparse::status_t> : std::true_type {};

namespace cusparse {
#define codes(transform) \
			transform(SUCCESS, "no error") \
			transform(NOT_INITIALIZED, "cusparse library initialization failed") \
			transform(ALLOC_FAILED, "resources could not be allocated") \
			transform(INVALID_VALUE, "invalid value") \
			transform(ARCH_MISMATCH, "architecture mismatch") \
			transform(MAPPING_ERROR, "mapping error") \
			transform(EXECUTION_FAILED, "execution failed") \
			transform(INTERNAL_ERROR, "internal error") \
			transform(ZERO_PIVOT, "the provided matrix yields a zero pivot") \
			transform(MATRIX_TYPE_NOT_SUPPORTED, "matrix type not supported")

	struct cusparse_category : std::error_category {
		virtual const char* name() const noexcept { return "cusparse"; }
		virtual std::string message(int condition) const
		{
			switch (static_cast<status_t>(condition)) {
#define message_transform(type, string) \
				case CUSPARSE_STATUS_##type: return string;
				codes(message_transform)
#undef message_transform
			}
			return "unknown error";
		}

		cusparse_category() = default;
	};
#undef codes

	inline std::error_code
	make_error_code(status_t e)
	{
		static const cusparse_category category;
		return {static_cast<int>(e), category};
	}

	struct exception : std::system_error {
		exception(status_t error) :
			std::system_error(make_error_code(error)) {}
		exception(status_t error, const std::string& what_arg) :
			std::system_error(make_error_code(error), what_arg) {}
	};

	inline bool is_success(status_t e) { return e == CUSPARSE_STATUS_SUCCESS; }
	inline bool is_failure(status_t e) { return !is_success(e); }

	inline void
	throw_if_error(status_t e)
	{
		if (is_failure(e)) throw exception(e);
	}

	inline void
	throw_if_error(status_t e, const std::string& what_arg)
	{
		if (is_failure(e)) throw exception(e, what_arg);
	}
}
