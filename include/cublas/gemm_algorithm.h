#pragma once
#include "util/adaptor.h"
#include "types.h"

namespace cublas {

#define CAT0(pre, post) pre##post
#define CAT(pre, post) CAT0(pre, post)
#define ALGO_PREFIX(n) CAT(algo, n)
#define TENSOR_ALGO_PREFIX(n) CAT(tensor_algo, n)
#define CUBLAS_ALGO_PRE(n) CAT(CUBLAS_GEMM_ALGO, n)
#define CUBLAS_TENSOR_ALGO_PRE(n) CAT(CAT(CUBLAS_GEMM_ALGO, n), _TENSOR_OP)
#define NEW_ALGO_PRE(n) CAT(gemm_algorithm::algo, n)
#define NEW_TENSOR_ALGO_PRE(n) CAT(gemm_algorithm::tensor_algo, n)
#define NON_TENSOR(transform) \
	transform(0), transform(1), transform(2), transform(3), \
	transform(4), transform(5), transform(6), transform(7), \
	transform(8), transform(9), transform(10), transform(11), \
	transform(12), transform(13), transform(14), transform(15), \
	transform(16), transform(17), transform(18), transform(19), \
	transform(20), transform(21), transform(22), transform(23)
#define TENSOR(transform) \
	transform(0), transform(1), transform(2), transform(3), \
	transform(4), transform(5), transform(6), transform(7), \
	transform(8), transform(9), transform(10), transform(11), \
	transform(12), transform(13), transform(14), transform(15)

enum class gemm_algorithm {
	default_algo,
	NON_TENSOR(ALGO_PREFIX),
	default_tensor_algo,
	TENSOR(TENSOR_ALGO_PREFIX)
};

using gemm_algo_adaptor = util::adaptor<
	util::enum_container<gemm_algo_t,
		CUBLAS_GEMM_DEFAULT,
		NON_TENSOR(CUBLAS_ALGO_PRE),
		CUBLAS_GEMM_DEFAULT_TENSOR_OP,
		TENSOR(CUBLAS_TENSOR_ALGO_PRE)>,
	util::enum_container<gemm_algorithm,
		gemm_algorithm::default_algo,
		NON_TENSOR(NEW_ALGO_PRE),
		gemm_algorithm::default_tensor_algo,
		TENSOR(NEW_TENSOR_ALGO_PRE)>>;

#undef CAT0
#undef CAT
#undef ALGO_PREFIX
#undef TENSOR_ALGO_PREFIX
#undef CUBLAS_ALGO_PRE
#undef CUBLAS_TENSOR_ALGO_PRE
#undef NEW_ALGO_PRE
#undef NEW_TENSOR_ALGO_PRE
#undef NON_TENSOR
#undef TENSOR

} // namespace cublas
