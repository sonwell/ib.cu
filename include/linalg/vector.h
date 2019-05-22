#pragma once
#include "base.h"
#include "dense.h"
#include "sparse.h"
#include "blasdefs.h"
#include "exceptions.h"

namespace linalg {

template <typename> class vector;

template <typename ltype, typename rtype>
vector<dense<decltype(ltype{} * rtype{})>>
hadamard(const vector<dense<ltype>>&, const vector<dense<rtype>>&);

template <template <typename> class layout, typename vtype>
class vector<layout<vtype>> : public layout<vtype> {
private:
	static_assert(is_field_v<vtype>, "type is not double, float, or complex");
protected:
	using super = layout<vtype>;
public:
	using index_type = typename super::index_type;
	using value_type = typename super::value_type;
private:
	static_assert(std::is_same_v<super, dense<value_type>> ||
			std::is_same_v<super, sparse<value_type>>,
			"vector only supports sparse and dense layouts");
protected:
	using super::copy;
	using super::swap;
public:
	template <typename otype>
	vector& operator=(const vector<layout<otype>>& o) { copy(o); return *this; }
	vector& operator=(const vector& o) { copy(o); return *this; }
	vector& operator=(vector&& o) { swap(o); return *this; }
	vector& operator+=(const vector& o) { axpy(1.0, o, *this); return *this; }
	vector& operator-=(const vector& o) { axpy(-1.0, o, *this); return *this; }
	vector& operator*=(value_type v) { scal(v, *this); return *this; }
	vector& operator/=(value_type v) { scal(1./v, *this); return *this; }
	vector& operator%=(const vector& o) { hadamard(o, *this); return *this; }

	template <typename ... arg_types>
	vector(size sz, arg_types&& ... args) :
		super(sz, std::forward<arg_types>(args)...)
	{ if (sz.cols != 1) throw mismatch("vector must have only 1 column"); }
	template <typename ... arg_types>
	vector(int rows, arg_types&& ... args) :
		vector(size{rows, 1}, std::forward<arg_types>(args)...) {}
	vector() : vector(size{0, 1}) {}
	vector(const vector& o) : vector() { copy(o); }
	vector(vector&& o) : vector() { swap(o); }
};

template <template <typename> class layout,
		 typename ltype, typename rtype>
inline auto
operator+(vector<layout<ltype>> left,
		const vector<layout<rtype>>& right)
{
	return left += right;
}

template <template <typename> class layout,
		 typename ltype, typename rtype>
inline auto
operator-(vector<layout<ltype>> left,
		const vector<layout<rtype>>& right)
{
	return left -= right;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator*(stype a, vector<layout<vtype>> v)
{
	return v *= a;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator*(vector<layout<vtype>> v, stype a)
{
	return v *= a;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator/(vector<layout<vtype>> v, stype a)
{
	return v /= a;
}


template <template <typename> class layout, typename vtype>
inline auto
operator%(vector<layout<vtype>> left, const vector<layout<vtype>>& right)
{
	return left %= right;
}

template <template <typename> class layout, typename vtype>
inline auto
operator-(vector<layout<vtype>> left)
{
	return left *= -1;
}

template <template <typename> class layout, typename vtype>
inline auto
operator+(vector<layout<vtype>> left)
{
	return left;
}

} // namespace linalg
