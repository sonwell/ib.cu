#pragma once
#include <stdexcept>
#include <string>
#include <cusparse.h>
#include <system_error>
#include "types.h"

template <> struct std::is_error_code_enum<cusparse::status_t> : std::true_type {};

namespace cusparse {

// An attempt to future-proof. More maintainable than macros, but now we have 3
// representations for the same data.
enum class status : std::underlying_type_t<status_t> {
	success                   = CUSPARSE_STATUS_SUCCESS,
	not_initialized           = CUSPARSE_STATUS_NOT_INITIALIZED,
	alloc_failed              = CUSPARSE_STATUS_ALLOC_FAILED,
	invalid_value             = CUSPARSE_STATUS_INVALID_VALUE,
	arch_mismatch             = CUSPARSE_STATUS_ARCH_MISMATCH,
	mapping_error             = CUSPARSE_STATUS_MAPPING_ERROR,
	execution_failed          = CUSPARSE_STATUS_EXECUTION_FAILED,
	internal_error            = CUSPARSE_STATUS_INTERNAL_ERROR,
	zero_pivot                = CUSPARSE_STATUS_ZERO_PIVOT,
	matrix_type_not_supported = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
#ifdef CUSPARSE_STATUS_NOT_SUPPORTED
	// Added in CUDA 10
	// Definitely need a better guard if they add more statuses.
	not_supported             = CUSPARSE_STATUS_NOT_SUPPORTED,
#else
	not_supported             = std::numeric_limits<std::underlying_type_t<status_t>>::max(),
#endif
};

struct cusparse_category : std::error_category {
	virtual const char* name() const noexcept { return "cusparse"; }
	virtual std::string message(int condition) const
	{
		switch (static_cast<status>(condition)) {
			case status::success:                   return "no error";
			case status::not_initialized:           return "cusparse library initialization failed";
			case status::alloc_failed:              return "resources could not be allocated";
			case status::invalid_value:             return "invalid value";
			case status::arch_mismatch:             return "architecture mismatch";
			case status::mapping_error:             return "mapping error";
			case status::execution_failed:          return "execution failed";
			case status::internal_error:            return "internal error";
			case status::zero_pivot:                return "the provided matrix yields a zero pivot";
			case status::matrix_type_not_supported: return "matrix type not supported";
			case status::not_supported:             return "not supported";
		}
		return "unknown error";
	}

	cusparse_category() = default;
};

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

inline bool is_success(status_t e) { return static_cast<status>(e) == status::success; }
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
