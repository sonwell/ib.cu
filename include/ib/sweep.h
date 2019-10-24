#pragma once
#include <array>
#include "util/math.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "delta.h"

namespace ib {
namespace __1 {

template <std::size_t n>
struct container {
	double values[n];

	constexpr double& operator[](int i) { return values[i]; }
	constexpr const double& operator[](int i) const { return values[i]; }

	constexpr container&
	operator+=(const container& o)
	{
		for (int i = 0; i < n; ++i)
			values[i] += o.values[i];
		return *this;
	}
};

template <std::size_t n>
constexpr container<n>
operator+(container<n> left, const container<n>& right)
{
	return left += right;
}

} // namespace __1

template <typename grid_type>
struct sweep {
	static constexpr auto dimensions = grid_type::dimensions;
	static constexpr auto nvalues = 1 << dimensions;
	static constexpr auto total_values = 1 << (2 * dimensions);
	static constexpr auto max_count = (total_values + nvalues - 1) / nvalues;
	static constexpr cosine_delta delta = {};
	using container_type = __1::container<nvalues>;

	int count;
	const grid_type& grid;

	static constexpr auto bit(int i, int n) { return (i & (1 << n)) >> n; }

	constexpr auto
	values(const fd::__1::delta<dimensions>& dx, double f) const
	{
		container_type values = {0.0};
		if (count >= max_count)
			return values;

		double weights[dimensions][2];
		for (int i = 0; i < dimensions; ++i) {
			auto shift = bit(count, i);
			auto y = dx[i] + shift - 1.0;
			auto v = delta(y);
			weights[i][0] = v;
			weights[i][1] = 0.5 - v;
		}

		for (int i = 0; i < nvalues; ++i) {
			double v = f;
			for (int j = 0; j < dimensions; ++j)
				v *= weights[j][bit(i, j)];
			values[i] = v;
		}

		return values;
	}

	constexpr auto
	indices(int index) const
	{
		std::array<int, nvalues> values = {0};

		int shifts[dimensions][2];
		for (int i = 0; i < dimensions; ++i) {
			auto shift = bit(count, i);
			shifts[i][0] = shift - 1;
			shifts[i][1] = shift + 1;
		}

		auto indices = grid.indices(index);
		for (int i = 0; i < nvalues; ++i) {
			fd::__1::shift<dimensions> s = {0};
			for (int j = 0; j < dimensions; ++j)
				s[j] = shifts[j][bit(i, j)];
			values[i] = grid.index(indices + s);
		}

		return values;
	}

	constexpr sweep(int count, const grid_type& grid) :
		count(count), grid(grid) {};
};

} // namespace ib
