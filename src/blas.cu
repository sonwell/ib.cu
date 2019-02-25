#include <iostream>
#include <cusparse.h>

#include "thrust/execution_policy.h"
#include "thrust/scan.h"
#include "thrust/inner_product.h"

#include "lwps/types.h"
#include "lwps/vector.h"
#include "lwps/matrix.h"

#include "cusparse/handle.h"
#include "cusparse/matrix.h"
#include "cusparse/operation.h"
#include "util/launch.h"
#include "cuda/exceptions.h"
#include "cuda/copy.h"

#include "device_ptr.h"


namespace lwps {
	void swap(vector& x, vector& y) { std::swap(x, y); }
	void swap(matrix& x, matrix& y) { std::swap(x, y); }
	void copy(const vector& x, vector& y) { y = x; }
	void copy(const matrix& x, matrix& y) { y = x; }

	void
	scal(value_type a, vector& x)
	{
		if (a == 1.0) return;
		auto elements = x.rows();
		auto* values = x.values();
		auto k = [=] __device__ (int tid) { values[tid] *= a; };
		util::transform<128, 7>(k, elements);
	}

	void
	scal(value_type a, matrix& x)
	{
		if (a == 1.0) return;
		auto elements = x.nonzero();
		auto* values = x.values();
		auto k = [=] __device__ (int tid) { values[tid] *= a; };
		util::transform<128, 7>(k, elements);
	}

	void
	axpy(value_type a, const vector& x, vector& y)
	{
		(void) (size(y) + a * size(x));
		auto* xvals = x.values();
		auto* yvals = y.values();
		auto k = [=] __device__ (int tid) { yvals[tid] += a * xvals[tid]; };
		util::transform<128, 7>(k, x.rows());
	}

	void
	axpy(value_type a, const matrix& x, matrix& y)
	{
		(void) (size(y) + a * size(x));
		auto rows = x.rows();
		auto cols = x.cols();
		if (x.nonzero() == 0) return;
		if (y.nonzero() == 0) { y = a * x; return; };

		mem::device_ptr<index_type> fstarts(rows+1);
		mem::device_ptr<index_type> index_buffer(x.nonzero() + y.nonzero());
		mem::device_ptr<value_type> value_buffer(x.nonzero() + y.nonzero());

		auto* xstarts = x.starts();
		auto* xindices = x.indices();
		auto* xvalues = x.values();
		auto* ystarts = y.starts();
		auto* yindices = y.indices();
		auto* yvalues = y.values();

		auto* tinds = index_buffer.data();
		auto* tvals = value_buffer.data();
		auto* starts = fstarts.data();

		auto compute = [=] __device__ (int tid)
		{
			auto xind = xstarts[tid];
			auto xend = xstarts[tid+1];
			auto yind = ystarts[tid];
			auto yend = ystarts[tid+1];
			auto start = xind + yind;

			index_type count = 0;

			auto xhas = xind < xend;
			auto yhas = yind < yend;
			bool xadv = true;
			bool yadv = true;
			auto xcol = cols;
			auto ycol = cols;
			value_type xval = 0;
			value_type yval = 0;

			while (xhas || yhas) {
				xcol = xadv ? xhas ? xindices[xind] : cols : xcol;
				xval = xadv ? xhas ? xvalues[xind] : 0 : xval;
				ycol = yadv ? yhas ? yindices[yind] : cols : ycol;
				yval = yadv ? yhas ? yvalues[yind] : 0 : yval;

				xadv = xcol <= ycol;
				yadv = ycol <= xcol;
				xind += xadv;
				yind += yadv;
				xhas = xind < xend;
				yhas = yind < yend;
				auto val = xadv * a * xval + yadv * yval;

				if (val) {
					tvals[start + count] = val;
					tinds[start + count] = xadv ? xcol : ycol;
				}
				count += val != 0;
			}

			starts[tid] = count;
			if (!tid) starts[rows] = 0;
		};
		util::transform(compute, rows);

		index_type nnz;
		thrust::execution_policy<thrust::system::cuda::tag> exec;
		auto end = thrust::exclusive_scan(exec, starts, starts+rows+1, starts);
		cuda::dtoh(&nnz, starts+rows, 1);

		if (nnz == x.nonzero() + y.nonzero()) {
			y = matrix{rows, cols, nnz, std::move(fstarts),
				std::move(index_buffer), std::move(value_buffer)};
			return;
		}

		mem::device_ptr<index_type> findices(nnz);
		mem::device_ptr<value_type> fvalues(nnz);
		auto* indices = findices.data();
		auto* values = fvalues.data();

		auto compress = [=] __device__ (int tid)
		{
			auto f = [&] ()
			{
				auto start = 0;
				auto end = rows+1;

				while (start != end) {
					auto middle = (start + end) >> 1;
					auto el = starts[middle];
					auto pred = el > tid;
					start = pred ? start : middle+1;
					end = pred ? middle : end;
				}
				return start-1;
			};

			auto row = f();
			auto start = starts[row];
			auto offset = xstarts[row] + ystarts[row];
			auto diff = tid - start;

			values[tid] = tvals[offset + diff];
			indices[tid] = tinds[offset + diff];
		};
		util::transform<128, 7>(compress, nnz);

		y = matrix{rows, cols, nnz, std::move(fstarts),
			std::move(findices), std::move(fvalues)};
	}

	double
	dot(const vector& x, const vector& y)
	{
		thrust::execution_policy<thrust::system::cuda::tag> exec;
		auto rows = x.rows();
		auto* xvals = x.values();
		auto* yvals = y.values();
		return thrust::inner_product(exec, xvals, xvals+rows, yvals, 0.);
	}

	void
	gemv(value_type a, const matrix& m, const vector& x, value_type b, vector& y)
	{
		if (m.nonzero() == 0 || !a) return scal(b, y);
		cusparse::handle handle;
		cusparse::matrix_description description;
		cusparse::operation_t operation = static_cast<cusparse::operation_t>(
				cusparse::operation::non_transpose);
		cusparseDcsrmv(handle, operation,
				m.rows(), m.cols(), m.nonzero(), &a,
				description, m.values(), m.starts(), m.indices(),
				x.values(), &b, y.values());
	}
}
