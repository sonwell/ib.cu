#pragma once
#include <iostream>
#include "dense.h"
#include "sparse.h"
#include "vector.h"
#include "blasdefs.h"

namespace linalg {

template <typename> class matrix;

template <typename vtype>
class matrix<sparse<vtype>> : public sparse<vtype> {
protected:
	using super = sparse<vtype>;
public:
	using value_type = typename super::value_type;
	using index_type = typename super::index_type;
protected:
	using index_ptr = typename super::index_ptr;
	using value_ptr = typename super::value_ptr;

	template <typename otype> void copy(const matrix<sparse<otype>>&);
	//template <typename otype> void copy(const matrix<dense<otype>>&);
	void swap(matrix&);
private:
	index_ptr _starts;
public:
	index_type* starts() const { return _starts.data(); }

	template <typename otype>
	matrix& operator=(const matrix<sparse<otype>>& o) { copy(o); return *this; }
	matrix& operator=(const matrix& o) { copy(o); return *this; }
	matrix& operator=(matrix&& o) { swap(o); return *this; }
	matrix& operator+=(const matrix& o) { axpy(1.0, o, *this); return *this; }
	matrix& operator-=(const matrix& o) { axpy(-1.0, o, *this); return *this; }
	matrix& operator*=(value_type v) { scal(v, *this); return *this; }
	matrix& operator/=(value_type v) { scal(1./v, *this); return *this; }

	matrix(size sz, int nnz, index_ptr starts, index_ptr indices, value_ptr values) :
		super(sz, nnz, move(indices), move(values)), _starts(move(starts)) {}
	matrix(int rows, int cols, int nnz, index_ptr s, index_ptr i, value_ptr v) :
		matrix({rows, cols}, nnz, move(s), move(i), move(v)) {}
	matrix(size sz, int nnz = 0) : matrix(sz, nnz, nnz ? sz.rows+1 : 0, nnz, nnz) {}
	matrix(int rows = 0, int cols = 0, int nnz = 0) : matrix(size{rows, cols}, nnz) {}
	template <typename otype> explicit matrix(const matrix<sparse<otype>>& o) :
		matrix() { copy(o); }
	//template <otype> explicit matrix(const matrix<dense<otype>>& o) :
	//	matrix() { copy(o); }
	matrix(const matrix& o) : matrix() { copy(o); }
	matrix(matrix&& o) : matrix() { swap(o); }
};

template <typename vtype>
class matrix<dense<vtype>> : public dense<vtype> {
protected:
	using super = dense<vtype>;
public:
	using value_type = typename super::value_type;
	using value_ptr = typename super::value_ptr;
protected:
	using super::copy;
	using super::swap;

	//template <typename otype> void copy(const matrix<sparse<otype>>&);
public:
	using super::rows;
	using super::cols;
	using super::values;

	template <typename otype>
	matrix& operator=(const matrix<dense<otype>>& o) { copy(o); return *this; }
	matrix& operator=(const matrix& o) { copy(o); return *this; }
	matrix& operator=(matrix&& o) { swap(o); return *this; }
	matrix& operator+=(const matrix& o) { axpy(1.0, o, *this); return *this; }
	matrix& operator-=(const matrix& o) { axpy(-1.0, o, *this); return *this; }
	matrix& operator*=(value_type v) { scal(v, *this); return *this; }
	matrix& operator/=(value_type v) { scal(1./v, *this); return *this; }
	matrix& operator%=(const matrix&);

	matrix(size sz, value_ptr values) : super(sz, move(values)) {}
	template <typename fill_type>
	matrix(size sz, const filler<fill_type>& f) : super(sz, f) {}
	matrix(int rows, int cols, value_ptr values) : matrix({rows, cols}, move(values)) {}
	template <typename fill_type>
	matrix(int rows, int cols, const filler<fill_type>& f) : matrix({rows, cols}, f) {}
	matrix(size sz) : matrix(sz, sz.rows * sz.cols) {}
	matrix(int rows = 0, int cols = 0) : matrix(size{rows, cols}) {}
	template <typename otype> explicit matrix(const matrix<dense<otype>>& o) :
		matrix() { copy(o); }
	//template <otype> explicit matrix(const matrix<sparse<otype>>& o) :
	//	matrix() { copy(o); }
	matrix(const matrix& o) : matrix() { copy(o); }
	matrix(matrix&& o) : matrix() { swap(o); }
};

template <typename vtype>
template <typename otype>
void
matrix<sparse<vtype>>::
copy(const matrix<sparse<otype>>& o)
{
	auto n  = o.nonzero() ? o.rows() + 1 : 0;
	index_ptr starts(n, _starts.get_allocator());

	auto* sdata = starts.data();
	auto* rdata = o.starts();
	auto k = [=] __device__ (int tid) { sdata[tid] = rdata[tid]; };
	util::transform<128, 8>(k, n);

	_starts = move(starts);
	super::operator=(o);
}

template <typename vtype>
void
matrix<sparse<vtype>>::
swap(matrix& o)
{
	std::swap(_starts, o._starts);
	super::operator=(std::move(o));
}

template <typename vtype>
matrix<dense<vtype>>&
matrix<dense<vtype>>::
operator%=(const matrix& o)
{
	(void) (size(*this) + size(o));

	auto n = rows() * cols();
	auto* vdata = values();
	auto* odata = o.values();
	auto k = [=] __device__ (int tid) { vdata[tid] *= odata[tid]; };
	util::transform<128, 8>(k, n);
	return *this;
}

template <template <typename> class layout,
		 typename ltype, typename rtype>
inline auto
operator+(matrix<layout<ltype>> left,
		const matrix<layout<rtype>>& right)
{
	//using result_type = decltype(ltype{} + rtype{});
	//matrix<layout<result_type>> tmp{left};
	return left += right;
}

template <template <typename> class layout,
		 typename ltype, typename rtype>
inline auto
operator-(matrix<layout<ltype>> left,
		const matrix<layout<rtype>>& right)
{
	//using result_type = decltype(ltype{} - rtype{});
	//matrix<layout<result_type>> tmp{left};
	return left -= right;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator*(stype a, matrix<layout<vtype>> m)
{
	//using result_type = decltype(a * vtype{});
	//matrix<layout<result_type>> tmp{m};
	return m *= a;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator*(matrix<layout<vtype>> m, stype a)
{
	return m *= a;
}

template <typename stype, template <typename> class layout,
		 typename vtype, typename = std::enable_if_t<is_scalar_v<stype>>>
inline auto
operator/(matrix<layout<vtype>> m, stype a)
{
	return m /= a;
}

template <template <typename> class layout, typename vtype>
inline auto
operator-(matrix<layout<vtype>> m)
{
	 return m *= -1;
}

template <template <typename> class layout, typename vtype>
inline auto
operator+(matrix<layout<vtype>> m)
{
	 return m;
}

template <template <typename> class mlayout,
		 template <typename> class vlayout, typename vtype>
inline auto
operator*(const matrix<mlayout<vtype>>& m, const vector<vlayout<vtype>>& v)
{
	(void) (size(m) * size(v));
	vector<dense<vtype>> tmp{m.rows()};
	gemv(1.0, m, v, 0.0, tmp);
	return tmp;
}

template <typename vtype>
inline auto
operator*(const matrix<sparse<vtype>>& left,
		const matrix<sparse<vtype>>& right)
{
	(void) (size(left) * size(right));
	matrix<sparse<vtype>> tmp{left.rows(), right.cols()};
	gemm(1.0, left, right, 0.0, tmp);
	return tmp;
}

template <template <typename> class llayout,
		 template <typename> class rlayout, typename vtype>
inline auto
operator*(const matrix<llayout<vtype>>& left,
		const matrix<rlayout<vtype>>& right)
{
	(void) (size(left) * size(right));
	matrix<dense<vtype>> tmp{left.rows(), right.cols()};
	gemm(1.0, left, right, 0.0, tmp);
	return tmp;
}

template <typename vtype>
inline auto
operator%(matrix<dense<vtype>> left,
		const matrix<dense<vtype>>& right)
{
	(void) (size(left) + size(right));
	return left %= right;
}

} // namespace linalg
