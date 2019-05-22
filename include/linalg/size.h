#pragma once
#include <type_traits>
#include <ostream>
#include "exceptions.h"

namespace linalg {

struct size { int rows, cols; };

constexpr bool
operator==(const size& left, const size& right)
{
	return left.rows == right.rows &&
		left.cols == right.cols;
}

constexpr size
operator+(const size& left, const size& right)
{
	constexpr const char* msg = "dimensions must match";
	return (left == right) ?  left :
		(throw mismatch(msg), size{0, 0});
}

constexpr size
operator-(const size& left, const size& right)
{
	return left + right;
}

template <typename value_type,
		 typename = std::enable_if_t<std::is_arithmetic_v<value_type>>>
constexpr size
operator*(value_type left, const size& right)
{
	return right;
}

template <typename value_type,
		 typename = std::enable_if_t<std::is_arithmetic_v<value_type>>>
constexpr size
operator*(const size& left, value_type right)
{
	return left;
}

template <typename value_type,
		 typename = std::enable_if_t<std::is_arithmetic_v<value_type>>>
constexpr size
operator/(const size& left, value_type right)
{
	return left;
}

constexpr size
operator*(const size& left, const size& right)
{
	constexpr const char* msg = "inner dimensions must match";
	return (left.cols == right.rows) ?
		size{left.rows, right.cols} :
		(throw mismatch(msg), size{0, 0});
}

std::ostream&
operator<<(std::ostream& out, const size& sz)
{
	return out << "[ " << sz.rows << " x " << sz.cols << " ]";
}

}
