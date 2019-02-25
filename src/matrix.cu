#include <algorithm>
#include <cmath>
#include <ostream>
#include <iomanip>
#include "lwps/matrix.h"
#include "lwps/blas.h"
#include "lwps/io.h"
#include "util/launch.h"
#include "cuda/copy.h"

namespace lwps {
	matrix::matrix() :
		matrix_base(0, 0),
		_nonzero(0),
		_starts(nullptr),
		_indices(nullptr),
		_values(nullptr) {}

	matrix::matrix(const matrix_base& base) :
		matrix_base(base),
		_nonzero(0),
		_starts(nullptr),
		_indices(nullptr),
		_values(nullptr) {}

	matrix::matrix(index_type rows, index_type cols) :
		matrix_base(rows, cols),
		_nonzero(0),
		_starts(nullptr),
		_indices(nullptr),
		_values(nullptr) {}

	matrix::matrix(const matrix_base& base, index_type nonzero) :
		matrix_base(base),
		_nonzero(nonzero),
		_starts(rows() && nonzero ? rows() + 1 : 0),
		_indices(nonzero),
		_values(nonzero) {}

	matrix::matrix(index_type rows, index_type cols, index_type nonzero) :
		matrix_base(rows, cols),
		_nonzero(nonzero),
		_starts(rows && nonzero ? rows + 1 : 0),
		_indices(nonzero),
		_values(nonzero) {}

	matrix::matrix(const matrix_base& base, index_type nonzero,
			index_ptr&& starts, index_ptr&& indices, value_ptr&& values) :
		matrix_base(base),
		_nonzero(nonzero),
		_starts(std::move(starts)),
		_indices(std::move(indices)),
		_values(std::move(values)) {}

	matrix::matrix(index_type rows, index_type cols, index_type nonzero,
			index_ptr&& starts, index_ptr&& indices, value_ptr&& values) :
		matrix_base(rows, cols),
		_nonzero(nonzero),
		_starts(std::move(starts)),
		_indices(std::move(indices)),
		_values(std::move(values)) {}

	matrix::matrix(const matrix& other) :
		matrix_base(other),
		_nonzero(other.nonzero()),
		_starts(other.rows() + 1),
		_indices(other.nonzero()),
		_values(other.nonzero())
	{ copy(other); }

	matrix::matrix(matrix&& other) :
		matrix_base((matrix_base&&) other),
		_nonzero(0),
		_starts(nullptr),
		_indices(nullptr),
		_values(nullptr)
	{ swap(other); }

	void
	matrix::copy(const matrix& other)
	{
		auto rows = other.rows();
		auto nnz = other.nonzero();

		auto* drow = starts();
		auto* dind = indices();
		auto* dval = values();

		auto* srow = other.starts();
		auto* sind = other.indices();
		auto* sval = other.values();

		auto k = [=] __device__ (int tid)
		{
			if (tid < rows + 1)
				drow[tid] = srow[tid];
			if (tid < nnz) {
				dind[tid] = sind[tid];
				dval[tid] = sval[tid];
			}
		};
		util::transform<128, 11>(k, std::max(rows+1, nnz));
	}

	void
	matrix::swap(matrix& other)
	{
		std::swap(_nonzero, other._nonzero);
		std::swap(_starts, other._starts);
		std::swap(_indices, other._indices);
		std::swap(_values, other._values);
	}

	matrix&
	matrix::operator=(const matrix& other)
	{
		return *this = matrix{other};
	}

	matrix&
	matrix::operator=(matrix&& other)
	{
		swap(other);
		matrix_base::operator=(std::move(other));
		return *this;
	}

	matrix&
	matrix::operator+=(const matrix& other)
	{
		axpy(1.0, other, *this);
		return *this;
	}

	matrix&
	matrix::operator-=(const matrix& other)
	{
		axpy(-1.0, other, *this);
		return *this;
	}

	matrix&
	matrix::operator*=(value_type k)
	{
		scal(k, *this);
		return *this;
	}

	matrix&
	matrix::operator/=(value_type k)
	{
		scal(1.0 / k, *this);
		return *this;
	}

	std::ostream&
	operator<<(std::ostream& out, const lwps::matrix& m)
	{
		auto* style = lwps::io::get_style();
		index_type* h_starts = new index_type[m.rows() + 1];
		index_type* h_indices = new index_type[m.nonzero()];
		value_type* h_values = new value_type[m.nonzero()];

		if (m.nonzero())
			cuda::dtoh(h_starts, m.starts(), m.rows() + 1);
		else
			std::memset(h_starts, 0, sizeof(index_type) * (m.rows() + 1));
		cuda::dtoh(h_indices, m.indices(), m.nonzero());
		cuda::dtoh(h_values, m.values(), m.nonzero());

		auto row_padding = style->begin_matrix.size() -
			style->begin_matrix_row.size();

		style->prepare(out);
		out << style->begin_matrix
			<< style->begin_matrix_row;
		for (auto row = 0; row < m.rows(); ++row) {
			auto&& end = h_starts[row+1];
			auto offset = h_starts[row];
			if (row) {
				out << style->end_matrix_row
					<< style->matrix_row_delimiter;
				for (auto i = 0; i < row_padding; ++i) out << ' ';
				out << style->begin_matrix_row;
			}
			for (auto col = 0; col < m.cols(); ++col) {
				if (col)
					out << style->matrix_entry_delimiter;
				if (offset < end && col == h_indices[offset])
					style->value(out, h_values[offset++]);
				else
					style->value(out, 0);
			}
			if (offset != end)
				out << " ...";
		}
		out << style->end_matrix_row
			<< style->end_matrix;

		delete[] h_starts;
		delete[] h_indices;
		delete[] h_values;

		return out;
	}
}
