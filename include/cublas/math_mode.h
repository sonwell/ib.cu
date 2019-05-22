
namespace cublas {

enum class math_mode { default_math, tensor_ops };
using math_mode_adaptor = util::adaptor<
	util::enum_container<math_t,
			CUBLAS_DEFAULT_MATH,
			CUBLAS_TENSOR_OP_MATH>,
	util::enum_container<math_mode,
			math_mode::default_math,
			math_mode::tensor_ops>>;

inline math_mode
get_math_mode(const handle_t& h)
{
	math_t mode;
	cublasGetMathMode(h, &mode);
	return math_mode_adaptor(mode);
}

inline void
set_math_mode(handle_t& h, math_mode_adaptor mode)
{
	cublasSetMathMode(h, mode);
}

} // namespace cublas
