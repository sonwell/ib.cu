#pragma once
#include <stdexcept>
#include <string>
#include <system_error>
#include <cublas_v2.h>
#include "types.h"

namespace std { template <> struct is_error_code_enum<cublas::status_t> : std::true_type {}; }

namespace cublas {

// An attempt to future-proof. More maintainable than macros, but now we have 3
// representations for the same data.
enum class status : std::underlying_type_t<status_t> {
	success          = CUBLAS_STATUS_SUCCESS,
	not_initialized  = CUBLAS_STATUS_NOT_INITIALIZED,
	alloc_failed     = CUBLAS_STATUS_ALLOC_FAILED,
	invalid_value    = CUBLAS_STATUS_INVALID_VALUE,
	arch_mismatch    = CUBLAS_STATUS_ARCH_MISMATCH,
	mapping_error    = CUBLAS_STATUS_MAPPING_ERROR,
	execution_failed = CUBLAS_STATUS_EXECUTION_FAILED,
	internal_error   = CUBLAS_STATUS_INTERNAL_ERROR,
	not_supported    = CUBLAS_STATUS_NOT_SUPPORTED,
	license_error    = CUBLAS_STATUS_LICENSE_ERROR,
};

struct cublas_category : std::error_category {
	virtual const char* name() const noexcept { return "cublas"; }
	virtual std::string message(int condition) const
	{
		switch (static_cast<status>(condition)) {
			case status::success:          return "no error";
			case status::not_initialized:  return "cublas library initialization failed";
			case status::alloc_failed:     return "resources could not be allocated";
			case status::invalid_value:    return "invalid value";
			case status::arch_mismatch:    return "architecture mismatch";
			case status::mapping_error:    return "mapping error";
			case status::execution_failed: return "execution failed";
			case status::internal_error:   return "internal error";
			case status::not_supported:    return "not supported";
			case status::license_error:    return "license error";
		}
		return "unknown error";
	}

	cublas_category() = default;
};

inline std::error_code
make_error_code(status_t e)
{
	static const cublas_category category;
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
