#pragma once
#include <cusparse.h>
#include "cuda/version.h"
#include "cuda/types.h"
#include "handle.h"
#include "exceptions.h"

namespace cusparse {

inline cuda::version
get_version(handle& h)
{
	int v;
	throw_if_error(cusparseGetVersion(h, &v));
	return {v / 1000, (v % 1000) / 10};
}

inline int
get_property(cuda::library_property prop)
{
	int v;
	using prop_t = cuda::library_property_t;
	prop_t p = static_cast<prop_t>(prop);
	throw_if_error(cusparseGetProperty(p, &v));
	return v;
}

static inline const cuda::version version {
	get_property(cuda::library_property::major),
	get_property(cuda::library_property::minor),
	get_property(cuda::library_property::patch)
};

} // namespace cusparse
