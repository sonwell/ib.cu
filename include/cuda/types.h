#pragma once
#include <type_traits>
#include <atomic>
#include <cuda.h>
#include <cuComplex.h>
#include <library_types.h>

namespace cuda {

using status_t = cudaError_t;
using stream_t = cudaStream_t;
using event_t = cudaEvent_t;

using compute_mode_t = cudaComputeMode;
using device_prop_t = cudaDeviceProp;

using data_type_t = cudaDataType_t;
using library_property_t = libraryPropertyType_t;

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

enum class library_property : std::underlying_type_t<library_property_t> {
	major = MAJOR_VERSION,
	minor = MINOR_VERSION,
	patch = PATCH_LEVEL
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

struct adl {};

template <typename wrapped_type, typename lookup = adl>
class type_wrapper {
public:
	using value_type = wrapped_type;
private:
	template <typename adl>
	struct adl_wrapper {
		wrapped_type data;

		operator const wrapped_type&() const { return data; }
		operator wrapped_type&() { return data; }

		bool operator==(const adl_wrapper& o) { return data == o.data; }
		bool operator!=(const adl_wrapper& o) { return !operator==(o); }

		template <typename ... arg_types>
		adl_wrapper(arg_types&& ... args) :
			data{std::forward<arg_types>(args)...} {}
	};

	void
	copy(const type_wrapper& other)
	{
		value = other.value;
		count = other.count;
		++*count;
	}

	void
	swap(type_wrapper& other)
	{
		std::swap(value, other.value);
		std::swap(count, other.count);
	}
protected:
	adl_wrapper<lookup> value;
	std::atomic_int* count;
	struct internal {};

	template <typename ... arg_types>
	type_wrapper(internal, arg_types&& ... args) :
		count(new std::atomic_int(1))
	{ create(value, std::forward<arg_types>(args)...); }
public:
	operator const value_type&() const { return value; }
	operator value_type&() { return value; }

	type_wrapper& operator=(const type_wrapper& o) { copy(o); return *this; }
	type_wrapper& operator=(type_wrapper&& o) { swap(o); return *this; }

	bool operator==(const type_wrapper& o) { return value == o.value; }
	bool operator!=(const type_wrapper& o) { return !operator==(o); }

	type_wrapper(const type_wrapper& o) : count(nullptr) { copy(o); }
	type_wrapper(type_wrapper&& o) : count(nullptr) { swap(o); }
	type_wrapper(wrapped_type v) :
		value(v), count(new std::atomic_int(1)) {}

	type_wrapper() : type_wrapper(internal{}) {}
	template <typename first_type, typename ... arg_types,
			 typename = std::enable_if_t<sizeof...(arg_types) ||
				 !std::is_base_of_v<type_wrapper, std::decay_t<first_type>>>>
	type_wrapper(first_type&& first, arg_types&& ... args) :
		type_wrapper(internal{}, std::forward<first_type>(first),
				std::forward<arg_types>(args)...) {}

	~type_wrapper() {
		if (count != nullptr && --*count <= 0) {
			destroy(value);
			delete count;
		}
	}
};

} // namespace cuda
