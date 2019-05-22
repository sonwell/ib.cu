#pragma once
#include <cusolverDn.h>
#include <cusolverSp.h>
#include "cuda/types.h"
#include "cublas/types.h"
#include "cusparse/types.h"

namespace cusolver {

namespace dense { using handle_t = cusolverDnHandle_t; }
namespace sparse { using handle_t = cusolverSpHandle_t; }

using status_t = cusolverStatus_t;
using eig_type_t = cusolverEigType_t;
using eig_mode_t = cusolverEigMode_t;
using csrqr_info_t = csrqrInfo_t;
using syevj_info_t = syevjInfo_t;
using gesvdj_info_t = gesvdjInfo_t;

using cuda::stream_t;
using cublas::operation_t;
using cublas::fill_mode_t;
using cusparse::mat_descr_t;

struct adl {};
template <typename value_type, typename lookup = adl>
using type_wrapper = cuda::type_wrapper<value_type, adl>;

} // namespace cusolver
