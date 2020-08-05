#pragma once
#include <stdexcept>
#include <string>
#include <cusolverSp.h>
#include <system_error>
#include "types.h"

namespace std { template <> struct is_error_code_enum<cusolver::status_t> : std::true_type {}; }

namespace cusolver {

// An attempt to future-proof. More maintainable than macros, but now we have 3
// representations for the same data.
enum class status : std::underlying_type_t<status_t> {
	success                   = CUSOLVER_STATUS_SUCCESS,
	not_initialized           = CUSOLVER_STATUS_NOT_INITIALIZED,
	alloc_failed              = CUSOLVER_STATUS_ALLOC_FAILED,
	invalid_value             = CUSOLVER_STATUS_INVALID_VALUE,
	arch_mismatch             = CUSOLVER_STATUS_ARCH_MISMATCH,
	mapping_error             = CUSOLVER_STATUS_MAPPING_ERROR,
	execution_failed          = CUSOLVER_STATUS_EXECUTION_FAILED,
	internal_error            = CUSOLVER_STATUS_INTERNAL_ERROR,
	zero_pivot                = CUSOLVER_STATUS_ZERO_PIVOT,
	not_supported             = CUSOLVER_STATUS_NOT_SUPPORTED,
	matrix_type_not_supported = CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
	invalid_license           = CUSOLVER_STATUS_INVALID_LICENSE,
};

struct cusolver_category : std::error_category {
	virtual const char* name() const noexcept { return "cusolver"; }
	virtual std::string message(int condition) const
	{
		switch (static_cast<status>(condition)) {
			case status::success:                   return "no error";
			case status::not_initialized:           return "cusolver library initialization failed";
			case status::alloc_failed:              return "resources could not be allocated";
			case status::invalid_value:             return "invalid value";
			case status::arch_mismatch:             return "architecture mismatch";
			case status::mapping_error:             return "mapping error";
			case status::execution_failed:          return "execution failed";
			case status::internal_error:            return "internal error";
			case status::zero_pivot:                return "the provided matrix yields a zero pivot";
			case status::not_supported:             return "not supported";
			case status::matrix_type_not_supported: return "matrix type not supported";
			case status::invalid_license:           return "invalid license";
		}
		return "unknown error";
	}

	cusolver_category() = default;
};

inline std::error_code
make_error_code(status_t e)
{
	static const cusolver_category category;
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

} // namespace cusolver
