#include <stdexcept>
#include <iostream>
#include "util/launch.h"
#include "util/memory.h"
#include "cuda/device.h"
#include "cuda/copy.h"
#include "cublas/geam.h"
#include "cublas/gemm.h"
#include "util/debug.h"

struct matrix {
	using value_ptr = util::memory<double>;
	int rows;
	int cols;
	value_ptr _values;
	double* values() const { return _values.data(); }

	void
	swap(matrix& o)
	{
		std::swap(rows, o.rows);
		std::swap(cols, o.cols);
		std::swap(_values, o._values);
	}

	matrix(int rows, int cols, value_ptr values) :
		rows(rows), cols(cols), _values(std::move(values)) {}
	matrix(int rows, int cols) :
		rows(rows), cols(cols), _values(rows * cols) {}
	matrix(matrix&& o) : matrix(0, 0, nullptr) { swap(o); }
	auto& operator=(matrix&& o) { swap(o); return *this; }
};

std::ostream&
operator<<(std::ostream& out, const matrix& m)
{
	auto rows = m.rows;
	auto cols = m.cols;
	auto* h_values = new double[rows * cols];
	cuda::dtoh(h_values, m.values(), rows * cols);

	for (int i = 0; i < rows; ++i) {
		if (i) out << '\n';
		for (int j = 0; j < cols; ++j) {
			if (j) out << ", ";
			out << h_values[j * m.rows + i];
		}
	}

	delete[] h_values;
	return out;
}

void
geam(const cublas::handle& h, double alpha, const matrix& a,
		double beta, const matrix& b, matrix& c)
{
	auto rows = a.rows, cols = a.cols;
	if (rows != b.rows || cols != b.cols)
		throw std::runtime_error("foo");
	matrix result{rows, cols};
	auto op = cublas::operation::non_transpose;
	cublas::geam(h, op, op, rows, cols,
			&alpha, a.values(), a.rows,
			&beta, b.values(), b.rows,
			result.values(), rows);
	c = std::move(result);
}

void
gemm(const cublas::handle& h, double alpha, const matrix& a,
		const matrix& b, double beta, matrix& c)
{
	auto rows = a.rows, cols = b.cols;
	if (a.cols != b.rows)
		throw std::runtime_error("foo");
	auto op = cublas::operation::non_transpose;
	cublas::gemm(h, op, op, rows, cols, a.cols, &alpha,
			a.values(), a.rows, b.values(), b.rows,
			&beta, c.values(), c.rows);
}

matrix
operator+(const matrix& a, const matrix& b)
{
	cublas::handle h;
	matrix c{0, 0};
	geam(h, 1.0, a, 1.0, b, c);
	return std::move(c);
}

matrix
operator-(const matrix& a, const matrix& b)
{
	cublas::handle h;
	matrix c{0, 0};
	geam(h, 1.0, a, -1.0, b, c);
	return std::move(c);
}

matrix
operator*(const matrix& a, const matrix& b)
{
	cublas::handle h;
	matrix c{a.rows, b.cols};
	gemm(h, 1.0, a, b, 0.0, c);
	return std::move(c);
}

template <typename fill_func_type>
void
fill(matrix& m, fill_func_type fn)
{
	auto* values = m.values();
	auto rows = m.rows;
	auto cols = m.cols;

	auto k = [=] __device__ (int tid, auto f)
	{
		auto i = tid % rows;
		auto j = tid / rows;
		values[tid] = f(i, j);
	};
	util::transform<128, 8>(k, rows * cols, fn);
}

int
main(void)
{
	auto& dev = cuda::default_device();
	util::set_default_resource(dev.memory());
	matrix a{10, 10};
	matrix b{10, 10};

	auto* avalues = a.values();
	auto* bvalues = b.values();

	fill(a, [] __device__ (int, int) { return 1.0; });
	fill(b, [] __device__ (int, int) { return 2.0; });
	auto&& c = a - b;
	auto&& d = a * b;
	std::cout << c << std::endl;
	std::cout << d << std::endl;
}
