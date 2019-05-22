#pragma once
//#include "util/adaptor.h"
#include "types.h"

namespace cusolver {

//enum class eigenvalue_solver { a_lb, ab_l, ba_l };
//using eigenvalue_solve_adaptor = util::adaptor<
//	util::enum_container<eig_type_t,
//			CUSOLVER_EIG_TYPE_1,
//			CUSOLVER_EIG_TYPE_2,
//			CUSOLVER_EIG_TYPE_3>,
//	util::enum_container<eigenvalue_solver,
//			eigenvalue_solver::a_lb,
//			eigenvalue_solver::ab_l,
//			eigenvalue_solver::ba_l>>;
enum class eigenvalue_solver : std::underlying_type_t<eig_type_t> {
	a_lb = CUSOLVER_EIG_TYPE_1,
	ab_l = CUSOLVER_EIG_TYPE_2,
	ba_l = CUSOLVER_EIG_TYPE_3
};

//enum class eigenvalue_mode { no_vector, vector };
//using eigenvalue_solve_adaptor = util::adaptor<
//	util::enum_container<eig_type_t,
//			CUSOLVER_EIG_MODE_NO_VECTOR,
//			CUSOLVER_EIG_MODE_VECTOR>,
//	util::enum_container<eigenvalue_mode,
//			eigenvalue_mode::no_vector,
//			eigenvalue_mode::vector>>;
enum class eigenvalue_mode : std::underlying_type_t<eig_mode_t> {
	vector = CUSOLVER_EIG_MODE_VECTOR,
	no_vector = CUSOLVER_EIG_MODE_NO_VECTOR
};

}
