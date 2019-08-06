#pragma once
#include <cuda_runtime.h>

namespace cuda {

enum class pass { host, device };

namespace runtime {

static constexpr struct version {
	int major;
	int minor;
	int patch;

	constexpr bool
	operator==(const version& v) const
	{
		return major == v.major && minor == v.minor && patch == v.patch;
	}

	constexpr bool
	operator>(const version& v) const
	{
		return major > v.major ||
			(major == v.major && (minor > v.minor ||
			(minor == v.minor && patch > v.patch)));
	}

	constexpr bool operator>=(const version& v) const { return operator==(v) || operator>(v); }
	constexpr bool operator<(const version& v) const { return !operator>=(v); }
	constexpr bool operator<=(const version& v) const { return operator==(v) || operator<(v); }
	constexpr bool operator!=(const version& v) const { return !operator==(v); }

	constexpr version(int version) :
		major(version / 1000),
		minor((version % 1000) / 10),
		patch(version % 10) {}
} version =
#ifdef CUDART_VERSION
CUDART_VERSION;
#else
0;
#endif

} // namespace runtime

enum class features : bool {
	compilation = runtime::version > 0,
	relaxed_constexpr =
#if (defined(__NVCC__) && defined(__CUDACC_RELAXED_CONSTEXPR__)) || defined(__clang__)
		true,
#else
		false,
#endif
	extended_lambda =
#if (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__)) || defined(__clang__)
		true
#else
		false
#endif
};

namespace compiler {

#ifdef __CUDA_ARCH__
static constexpr cuda::pass pass = cuda::pass::device;
#else
static constexpr cuda::pass pass = cuda::pass::host;
#endif

#if defined(__NVCC__) || defined(__clang__)
static constexpr auto can_compile_cuda = true;
#else
static constexpr auto can_compile_cuda = false;
#endif


template <features feature>
inline constexpr bool supports = can_compile_cuda && static_cast<bool>(feature);

} // namespace compiler
} // namespace cuda
