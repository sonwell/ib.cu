#pragma once
#include "cublas/handle.h"
#include "cublas/operation.h"
#include "cublas/scal.h"
#include "cublas/axpy.h"
#include "cublas/dot.h"
#include "cublas/nrm2.h"
#include "cublas/gemv.h"
#include "cublas/gemm.h"

#include "cusparse/handle.h"
#include "cusparse/matrix_descr.h"
#include "cusparse/operation.h"
#include "cusparse/index_base.h"
#include "cusparse/axpyi.h"
#include "cusparse/doti.h"
#include "cusparse/csrmv.h"
#include "cusparse/gemvi.h"
#include "cusparse/csrmm.h"
#include "cusparse/csrgeam.h"
#include "cusparse/csrgemm.h"

#include "types.h"
#include "blasdefs.h"
#include "vector.h"
#include "matrix.h"
#include "exceptions.h"
#include "io.h"
#include "fill.h"

namespace linalg {

template <template <typename> class container, typename vtype>
int
length(const container<sparse<vtype>>& x)
{
	return x.nonzero();
}

template <template <typename> class container, typename vtype>
int
length(const container<dense<vtype>>& x)
{
	return x.rows() * x.cols();
}

template <template <typename> class container,
		 template <typename> class layout, typename vtype>
void
swap(container<layout<vtype>>& x, container<layout<vtype>>& y)
{
	std::swap(x, y);
}

template <template <typename> class container,
		 template <typename> class layout, typename ftype, typename ttype>
void
copy(const container<layout<ftype>>& x, container<layout<ttype>>& y)
{
	y = x;
}

template <template <typename> class container, typename vtype>
void
scal(scalar<vtype> s, container<dense<vtype>>& y)
{
	if (s == 1 || !length(y)) return;
	if (!s) return (void) (y = container<dense<vtype>>{size(y), zero});
	auto n = length(y);
	auto* values = y.values();
	auto k = [=] __device__ (int tid) { values[tid] *= s; };
	util::transform<128, 7>(k, n);
}

template <template <typename> class container, typename vtype>
void
scal(scalar<vtype> s, container<sparse<vtype>>& y)
{
	if (s == 1 || !length(y)) return;
	if (!s) return (void) (y = container<sparse<vtype>>{size(y)});
	auto n = length(y);
	auto* values = y.values();
	auto k = [=] __device__ (int tid) { values[tid] *= s; };
	util::transform<128, 7>(k, n);
}

template <template <typename> class container, typename vtype>
void
axpy(scalar<vtype> s, const container<dense<vtype>>& x, container<dense<vtype>>& y)
{
	(void) (s * size(x) + size(y));
	if (!s) return;
	auto n = length(y);
	auto* xdata = x.values();
	auto* ydata = y.values();
	auto k = [=] __device__ (int tid) { ydata[tid] += s * xdata[tid]; };
	util::transform<128, 7>(k, n);
}

template <typename vtype>
void
axpy(scalar<vtype> s, const matrix<sparse<vtype>>& x, matrix<sparse<vtype>>& y)
{
	using matrix_type = matrix<sparse<vtype>>;
	if (!s || !x.nonzero()) return;
	static constexpr double beta = 1.0;
	(void) (s * size(x) + size(y));
	if (!y.nonzero()) return void (y = s * x);

	int nnz;
	util::memory<int> starts(x.rows() + 1);
	cusparse::handle h;
	cusparse::matrix_description descr;

	cusparse::csrgeam_nnz(h, x.rows(), x.cols(),
			descr, x.nonzero(), x.starts(), x.indices(),
			descr, y.nonzero(), y.starts(), y.indices(),
			descr, starts.data(), &nnz);
	util::memory<vtype> values(nnz);
	util::memory<int> indices(nnz);
	cusparse::csrgeam(h, x.rows(), x.cols(),
			&s, descr, x.nonzero(), x.values(), x.starts(), x.indices(),
			&beta, descr, y.nonzero(), y.values(), y.starts(), y.indices(),
			descr, values.data(), starts.data(), indices.data());
	matrix_type result{x.rows(), x.cols(), nnz,
			std::move(starts), std::move(indices), std::move(values)};
	y = std::move(result);
}

template <typename vtype>
void
axpy(scalar<vtype> s, const vector<sparse<vtype>>& x, vector<dense<vtype>>& y)
{
	(void) (s * size(x) + size(y));
	cusparse::handle h;
	cusparse::axpyi(h, x.nonzero(), x.values(), x.indices(),
			y.values(), index_base);
}

template <template <typename> class layout, typename vtype>
vtype
dot(const vector<layout<vtype>>& x, const vector<layout<vtype>>& y)
{
	(void) (size(x) + size(y));
	vtype result;
	cublas::handle h;
	cublas::dot(h, length(x), x.values(), 1, y.values(), 1, &result);
	return result;
}

template <typename vtype>
vtype
dot(const vector<sparse<vtype>>& x, const vector<dense<vtype>>& y)
{
	(void) (size(x) + size(y));
	vtype result;
	cusparse::handle h;
	cusparse::doti(h, length(x), x.values(), x.indices(),
			y.values(), &result, index_base);
	return result;
}

template <template <typename> class layout, typename vtype>
vtype
nrm2(const vector<layout<vtype>>& x)
{
	base_t<vtype> result;
	cublas::handle h;
	cublas::nrm2(h, length(x), x.values(), 1, &result);
	return result;
}

template <typename vtype>
void
gemv(scalar<vtype> alpha, const matrix<dense<vtype>>& a, const vector<dense<vtype>>& x,
		scalar<vtype> beta, vector<dense<vtype>>& y)
{
	(void) (alpha * size(a) * size(x) + beta * size(y));
	if (!alpha) return scal(beta, y);
	cublas::handle h;
	cublas::operation op = cublas::operation::non_transpose;
	cublas::gemv(h, op, a.rows(), a.cols(), &alpha, a.values(), a.rows(),
			x.values(), 1, &beta, y.values(), 1);
}

template <typename vtype>
void
gemv(scalar<vtype> alpha, const matrix<sparse<vtype>>& a, const vector<dense<vtype>>& x,
		scalar<vtype> beta, vector<dense<vtype>>& y)
{
	(void) (alpha * size(a) * size(x) + beta * size(y));
	if (!alpha || !a.nonzero()) return scal(beta, y);
	cusparse::handle h;
	cusparse::operation op = cusparse::operation::non_transpose;
	cusparse::matrix_description descr;
	cusparse::csrmv(h, op, a.rows(), a.cols(), a.nonzero(), &alpha,
			descr, a.values(), a.starts(), a.indices(), x.values(),
			&beta, y.values());
}

template <typename vtype>
void
gemv(scalar<vtype> alpha, const matrix<dense<vtype>>& a, const vector<sparse<vtype>>& x,
		scalar<vtype> beta, vector<dense<vtype>>& y)
{
	if (!alpha) return scal(beta, y);
	(void) (alpha * size(a) * size(x) + beta * size(y));
	int buffer_size;
	cusparse::handle h;
	cusparse::operation op = cusparse::operation::non_transpose;
	cusparse::gemvi_buffer_size(h, op, a.rows(), a.cols(), a.nonzero(), &buffer_size);

	util::memory<void> buffer(buffer_size);
	cusparse::gemvi(h, op, a.rows(), a.cols(), &alpha,
			a.values(), a.rows(), x.values(), x.indices(), &beta,
			y.values(), index_base, buffer);
}

template <typename vtype>
void
gemm(scalar<vtype> alpha, const matrix<dense<vtype>>& a, const matrix<dense<vtype>>& b,
		scalar<vtype> beta, matrix<dense<vtype>>& c)
{
	if (!alpha) return scal(beta, c);
	(void) (alpha * size(a) * size(b) + beta * size(c));
	cublas::handle h;
	cublas::operation op = cublas::operation::non_transpose;
	cublas::gemm(h, op, op, a.rows(), b.cols(), a.cols(),
			&alpha, a.values(), a.rows(), b.values(), b.rows(),
			&beta, c.values(), c.rows());
}

template <typename vtype>
void
gemm(scalar<vtype> alpha, const matrix<sparse<vtype>>& a, const matrix<dense<vtype>>& b,
		scalar<vtype> beta, matrix<dense<vtype>>& c)
{
	if (!alpha) return scal(beta, c);
	(void) (alpha * size(a) * size(b)  + beta * size(c));
	cusparse::handle h;
	cusparse::operation op = cublas::operation::non_transpose;
	cusparse::matrix_description descr;
	cusparse::csrmm(h, op, a.rows(), b.cols(), a.cols(),
			&alpha, a.values(), a.starts(), a.indices(),
			b.values(), b.rows(), &beta, c.values(), c.rows());
}

template <typename vtype>
void
gemm(scalar<vtype> alpha, const matrix<sparse<vtype>>& a, const matrix<sparse<vtype>>& b,
		scalar<vtype> beta, matrix<sparse<vtype>>& c)
{
	if (!alpha) return scal(beta, c);
	(void) (alpha * size(a) * size(b) + beta * size(c));

	int nnz;
	cusparse::handle h;
	cusparse::operation op = cusparse::operation::non_transpose;
	cusparse::matrix_description descr;
	util::memory<int> starts(a.rows() + 1);
	cusparse::csrgemm_nnz(h, op, op, a.rows(), b.cols(), a.cols(),
			descr, a.nonzero(), a.starts(), a.indices(),
			descr, b.nonzero(), b.starts(), b.indices(),
			descr, starts, &nnz);
	if (!nnz)
		return (void) (c = matrix<sparse<vtype>>{});
	util::memory<vtype> values(nnz);
	util::memory<int> indices(nnz);
	cusparse::csrgemm(h, op, op, a.rows(), b.cols(), a.cols(),
			descr, a.nonzero(), a.values(), a.starts(), a.indices(),
			descr, b.noznero(), b.values(), b.starts(), b.indices(),
			descr, values, starts, indices);
	matrix<sparse<vtype>> tmp{a.rows(), b.cols(), nnz,
		std::move(starts), std::move(indices), std::move(values)};
	scal(beta, c);
	axpy(alpha, tmp, c);
}

template <template <typename> class container, typename vtype>
void
hadamard(const container<dense<vtype>>& x, container<dense<vtype>>& y)
{
	(void) (size(x) + size(y));

	auto n = x.rows() * x.cols();
	auto* xdata = x.values();
	auto* ydata = y.values();
	auto k = [=] __device__ (int tid) { ydata[tid] *= xdata[tid]; };
	util::transform<128, 8>(k, n);
}

template <template <typename> class container, typename vtype>
void
hadamard(const container<sparse<vtype>>& x, container<sparse<vtype>>& y)
{
	throw not_implemented("hadamard product is not defined for sparse containers");
}

}
