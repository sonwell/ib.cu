#pragma once
#include <type_traits>
#include <cuda.h>
#include <library_types.h>
#include <cuComplex.h>

namespace cuda {
	using status_t = cudaError_t;
	using stream_t = cudaStream_t;
	using event_t = cudaEvent_t;

	using compute_mode_t = cudaComputeMode;
	using device_prop_t = cudaDeviceProp;

	using data_type_t = cudaDataType_t;

	enum class data_type : std::underlying_type_t<data_type_t> {
		half_precision_real = CUDA_R_16F,
		single_precision_real = CUDA_R_32F,
		double_precision_real = CUDA_R_64F,
		half_precision_complex = CUDA_C_16F,
		single_precision_complex = CUDA_C_32F,
		double_precision_complex = CUDA_C_64F,
		signed_char_real = CUDA_R_8I,
		signed_char_complex = CUDA_C_8I,
		unsigned_char_real = CUDA_R_8U,
		unsigned_char_complex = CUDA_C_8U
	};

	template <typename> struct library_type;
	template <> struct library_type<double> :
		std::integral_constant<data_type, data_type::double_precision_real> {};
	template <> struct library_type<float> :
		std::integral_constant<data_type, data_type::single_precision_real> {};
	template <> struct library_type<cuDoubleComplex> :
		std::integral_constant<data_type, data_type::double_precision_complex> {};
	template <> struct library_type<cuFloatComplex> :
		std::integral_constant<data_type, data_type::single_precision_complex> {};
	template <typename type> inline bool library_type_v = library_type<type>::value;

	template <typename> struct is_numerical_type : std::false_type {};
	template <> struct is_numerical_type<double> : std::true_type {};
	template <> struct is_numerical_type<float> : std::true_type {};
	template <> struct is_numerical_type<cuDoubleComplex> : std::true_type {};
	template <> struct is_numerical_type<cuFloatComplex> : std::true_type {};
}
