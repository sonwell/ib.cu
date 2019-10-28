#pragma once
#include "linalg/linalg.h"
#include "fd/grid.h"

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

using fd::__1::delta;
using fd::__1::shift;

} // namespace ib
