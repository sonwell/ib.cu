#pragma once
#include <system_error>
#include <cuda.h>
#include "types.h"
namespace std { template <> struct is_error_code_enum<cuda::status_t> : std::true_type {}; }

namespace cuda {
	struct cuda_category : std::error_category {
		virtual const char* name() const noexcept { return "cuda"; }
		virtual std::string message(int condition) const
		{
			return cudaGetErrorString(static_cast<status_t>(condition));
		}

		cuda_category() = default;
	};

	inline std::error_code
	make_error_code(status_t e)
	{
		return {static_cast<int>(e), cuda_category()};
	}

	struct exception : std::system_error {
		exception(status_t error) :
			std::system_error(make_error_code(error)) {}
		exception(status_t error, const std::string& what_arg) :
			std::system_error(make_error_code(error), what_arg) {}
	};

	inline bool is_success(status_t e) { return e == cudaSuccess; }
	inline bool is_failure(status_t e) { return !is_success(e); }

	inline void
	throw_if_error(status_t result)
	{
		status_t e = static_cast<status_t>(result);
		if (is_failure(e)) throw exception(e);
	}

	inline void
	throw_if_error(status_t result, const std::string& what_arg)
	{
		status_t e = static_cast<status_t>(result);
		if (is_failure(e)) throw exception(e, what_arg);
	}
}
