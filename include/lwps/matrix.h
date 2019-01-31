#pragma once
#include "base.h"
#include "device_ptr.h"
#include "blas.h"
#include "vector.h"

namespace lwps {
	class matrix : public matrix_base {
		private:
			using index_ptr = mem::device_ptr<index_type>;
			using value_ptr = mem::device_ptr<value_type>;
		protected:
			index_type _nonzero;
			index_ptr _starts;
			index_ptr _indices;
			value_ptr _values;
		public:
			index_type nonzero() const { return _nonzero; }
			index_type* starts() { return _starts; }
			const index_type* starts() const { return _starts; }
			index_type* indices() { return _indices; }
			const index_type* indices() const { return _indices; }
			value_type* values() { return _values; }
			const value_type* values() const { return _values; }

			void copy(const matrix&);
			void swap(matrix&);

			matrix();
			matrix(const matrix_base&);
			matrix(index_type, index_type);
			matrix(const matrix_base&, index_type);
			matrix(index_type, index_type, index_type);
			matrix(const matrix_base&, index_type, index_ptr&&, index_ptr&&, value_ptr&&);
			matrix(index_type, index_type, index_type, index_ptr&&, index_ptr&&, value_ptr&&);
			matrix(const matrix&);
			matrix(matrix&&);
			matrix(const matrix_size& size) : matrix(size.rows, size.cols) {}

			matrix& operator=(const matrix&);
			matrix& operator=(matrix&&);
			matrix& operator+=(const matrix&);
			matrix& operator-=(const matrix&);
			matrix& operator*=(value_type);
			matrix& operator/=(value_type);
	};

	std::ostream& operator<<(std::ostream&, const matrix&);

	inline matrix
	operator+(matrix left, const matrix& right)
	{ return std::move(left += right); }

	inline matrix
	operator-(matrix left, const matrix& right)
	{ return std::move(left -= right); }

	inline matrix
	operator*(value_type a, matrix m)
	{ return std::move(m *= a); }

	inline matrix
	operator*(matrix m, value_type a)
	{ return std::move(m *= a); }

	inline matrix
	operator/(value_type a, matrix m)
	{ return std::move(m /= a); }

	inline matrix
	operator/(matrix m, value_type a)
	{ return std::move(m /= a); }

	inline matrix
	operator-(matrix m)
	{ return std::move(m *= -1); }

	inline matrix
	operator+(matrix m)
	{ return std::move(m); }

	inline vector
	operator*(const matrix& m, const vector& v)
	{
		(void) (size(m) * size(v));
		vector result{m.rows()};
		gemv(1.0, m, v, 0.0, result);
		return std::move(result);
	}
}
