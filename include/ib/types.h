#pragma once
#include "util/wrapper.h"
#include "linalg/dense.h"
#include "linalg/matrix.h"
#include "linalg/vector.h"

namespace ib {

using dense = linalg::dense<double>;
using matrix = linalg::matrix<dense>;
using vector = linalg::vector<dense>;

template <std::size_t n>
struct values_container {
	double values[n];

	constexpr double& operator[](int i) { return values[i]; }
	constexpr const double& operator[](int i) const { return values[i]; }

	constexpr values_container&
	operator+=(const values_container& o)
	{
		for (int i = 0; i < n; ++i)
			values[i] += o.values[i];
		return *this;
	}
};

template <std::size_t n>
constexpr values_container<n>
operator+(values_container<n> l, const values_container<n>& r)
{
	l += r;
	return l;
}

template <std::size_t n> using difference =
	util::wrapper<struct difference_tag, std::array<double, n>>;
template <std::size_t n> using point =
	util::wrapper<struct point_tag, std::array<double, n>>;
template <std::size_t n> using indices =
	util::wrapper<struct indices_tag, std::array<int, n>>;


} // namespace ib
