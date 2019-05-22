#pragma once
#include <cublas_v2.h>
#include "cuda/types.h"

namespace cublas {

using handle_t = cublasHandle_t;
using status_t = cublasStatus_t;
using operation_t = cublasOperation_t;
using fill_mode_t = cublasFillMode_t;
using diag_type_t = cublasDiagType_t;
using side_mode_t = cublasSideMode_t;
using pointer_mode_t = cublasPointerMode_t;
using atomics_mode_t = cublasAtomicsMode_t;
using gemm_algo_t = cublasGemmAlgo_t;
using math_t = cublasMath_t;

using cuda::stream_t;

struct adl {};
template <typename value_type, typename lookup = adl>
using type_wrapper = cuda::type_wrapper<value_type, adl>;

} // namespace cublas
