#pragma once
#include "types.h"
#include "handle.h"

namespace cusparse {

inline void
create_identity_permutation(handle& h, int n, int* p)
{
	throw_if_error(cusparseCreateIdentityPermutationx(h, n, p));
}

} // namespace cusparse