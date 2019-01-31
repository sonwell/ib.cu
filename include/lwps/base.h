#pragma once
#include <stdexcept>
#include <ostream>
#include "namespace.h"
#include "types.h"

namespace LWPS_NAMESPACE {
	class size_mismatch_error : public std::runtime_error {
		public:
			size_mismatch_error(const char* what) noexcept :
				std::runtime_error(what) {}
	};

	struct matrix_size {
		index_type rows;
		index_type cols;

		constexpr bool operator==(const matrix_size&) const;
		constexpr bool operator!=(const matrix_size& other) const { return !(*this == other); }

		constexpr matrix_size operator+(const matrix_size&) const;
		constexpr matrix_size operator-(const matrix_size&) const;
		constexpr matrix_size operator%(const matrix_size&) const;
		constexpr matrix_size operator*(const matrix_size&) const;
		constexpr matrix_size operator*(value_type) const { return *this; }
		constexpr matrix_size operator/(value_type) const { return *this; }
	};

	constexpr bool
	matrix_size::operator==(const matrix_size& other) const
	{
		return rows == other.rows && cols == other.cols;
	}

	constexpr matrix_size
	matrix_size::operator+(const matrix_size& other) const
	{
		return operator==(other) ? *this :
			(throw size_mismatch_error("Dimensions must match for addition"), *this);
	}

	constexpr matrix_size
	matrix_size::operator-(const matrix_size& other) const
	{
		return operator==(other) ? *this :
			(throw size_mismatch_error("Dimensions must match for subtraction"), *this);
	}

	constexpr matrix_size
	matrix_size::operator%(const matrix_size& other) const
	{
		return operator==(other) ? *this :
			(throw size_mismatch_error("Dimensions must match for element-wise product"), *this);
	}

	constexpr matrix_size
	matrix_size::operator*(const matrix_size& other) const
	{
		return cols == other.rows ? matrix_size{rows, other.cols} :
			(throw size_mismatch_error("Inner dimensions must match for multiplication"), *this);
	}

	constexpr matrix_size operator*(value_type, const matrix_size& b) { return b; }

	inline std::ostream&
	operator<<(std::ostream& out, const matrix_size& size)
	{
		return out << size.rows << ' ' << size.cols;
	}

	struct matrix_base {
	private:
		matrix_size _size;
	public:
		constexpr const index_type& rows() const { return _size.rows; }
		constexpr const index_type& cols() const { return _size.cols; }

		matrix_base& operator=(const matrix_base&) = default;

		constexpr matrix_base(index_type rows, index_type cols) :
			_size{rows, cols} {}

		friend constexpr matrix_size size(const matrix_base&);
	};

	constexpr matrix_size size(const matrix_base& m) { return m._size; }
}
