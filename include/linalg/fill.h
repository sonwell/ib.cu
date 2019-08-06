#pragma once
#include <type_traits>
#include "types.h"
#include "dense.h"
#include "util/functional.h"

namespace linalg {

template <typename fill_type, bool>
class filler {
private:
	fill_type fill;
public:
	template <typename ... arg_types>
	constexpr auto
	operator()(arg_types&& ... args) const
	{
		return fill(std::forward<arg_types>(args)...);
	}

	constexpr filler(fill_type f) : fill(f) {}
};

template <typename value_type>
class filler<value_type, true> {
private:
	value_type value;
public:
	template <typename cast_type, typename = std::enable_if_t<
			 std::is_convertible_v<value_type, cast_type>>>
	constexpr explicit operator cast_type() { return value; }
	template <typename cast_type, typename = std::enable_if_t<
			 std::is_convertible_v<value_type, cast_type>>>
	constexpr explicit operator filler<cast_type>()
	{ return filler<cast_type>(value); }

	template <typename ... arg_types>
	constexpr auto operator()(arg_types&&...) { return value; }

	constexpr filler(value_type value) : value(value) {}
};

template <typename value_type>
filler(value_type) -> filler<value_type>;

template <typename value_type,
		 typename = std::enable_if_t<is_scalar_v<value_type>>>
struct constant : filler<value_type> {
	using filler<value_type>::filler;
};

template <typename value_type>
constant(value_type) -> constant<value_type>;

inline constexpr constant one(1.0);
inline constexpr constant zero(0.0);

template <typename rtype, typename ltype>
constexpr auto
operator+(const constant<rtype>& r, const constant<ltype>& l)
{
	return constant((rtype) r + (ltype) l);
}

template <typename rtype, typename ltype>
constexpr auto
operator-(const constant<rtype>& r, const constant<ltype>& l)
{
	return constant((rtype) r - (ltype) l);
}

template <typename rtype, typename ltype>
constexpr auto
operator*(const constant<rtype>& r, const constant<ltype>& l)
{
	return constant((rtype) r * (ltype) l);
}

template <typename rtype, typename ltype>
constexpr auto
operator/(const constant<rtype>& r, const constant<ltype>& l)
{
	return constant((rtype) r / (ltype) l);
}


template <typename vtype, typename fn_type,
		 typename = std::enable_if_t<!is_scalar_v<fn_type>>>
void
fill(dense<vtype>& m, fn_type fn)
{
	using namespace util::functional;
	auto rows = m.rows();
	auto cols = m.cols();
	auto n = rows * cols;
	auto* mdata = m.values();

	auto k = [=] __device__ (int tid, auto fn)
	{
		mdata[tid] = fn(tid);
	};
	util::transform<128, 8>(k, n, fn);
}

template <template <typename> class container, typename vtype>
void
fill(container<dense<vtype>>& x, scalar<vtype> v)
{
	fill(x, constant{v});
}

template <typename vtype>
constexpr auto
fill(vtype v)
{
	return filler{v};
}

} // namespace linalg
