#pragma once
#include "util/adaptor.h"
#include "types.h"

namespace cublas {

enum class atomics { not_allowed, allowed };
using atomics_adaptor = util::adaptor<
	util::enum_container<atomics_mode_t,
			CUBLAS_ATOMICS_NOT_ALLOWED,
			CUBLAS_ATOMICS_ALLOWED>,
	util::enum_container<atomics,
			atomics::not_allowed,
			atomics::allowed>>;

inline atomics
get_atomics_mode(const handle_t& h)
{
	atomics_mode_t mode;
	cublasGetAtomicsMode(h, &mode);
	return static_cast<atomics>(mode);
}

inline void
set_atomics_mode(handle_t& h, atomics_adaptor mode)
{
	cublasSetAtomicsMode(h, mode);
}

} // namespace cublas
