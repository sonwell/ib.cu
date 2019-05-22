#pragma once
#include <cusparse.h>
#include "cuda/types.h"

namespace cusparse {

using status_t = cusparseStatus_t;
using handle_t = cusparseHandle_t;
using hyb_mat_t = cusparseHybMat_t;
using mat_descr_t = cusparseMatDescr_t;
using solve_analysis_info_t = cusparseSolveAnalysisInfo_t;

using action_t = cusparseAction_t;
using algorithm_mode_t = cusparseAlgMode_t;
using diagonal_type_t = cusparseDiagType_t;
using fill_mode_t = cusparseFillMode_t;
using index_base_t = cusparseIndexBase_t;
using matrix_type_t = cusparseMatrixType_t;
using hyb_partition_t = cusparseHybPartition_t;
using direction_t = cusparseDirection_t;
using operation_t = cusparseOperation_t;
using pointer_mode_t = cusparsePointerMode_t;
using solve_policy_y = cusparseSolvePolicy_t;
using color_info_t = cusparseColorInfo_t;

using bsric02_info_t = bsric02Info_t;
using bsrilu02_info_t = bsrilu02Info_t;
using bsrsm2_info_t = bsrsm2Info_t;
using bsrsv2_info_t = bsrsv2Info_t;
using csrgemm2_info_t = csrgemm2Info_t;
using csric02_info_t = csric02Info_t;
using csrilu02_info_t = csrilu02Info_t;
using csrsm2_info_t = csrsm2Info_t;
using csrsv2_info_t = csrsv2Info_t;
using prune_info_t = pruneInfo_t;

using cuda::stream_t;

struct adl {};
template <typename value_type, typename lookup = adl>
using type_wrapper = cuda::type_wrapper<value_type, adl>;

} // namespace cusparse
