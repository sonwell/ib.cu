#pragma once
#include "util/math.h"
#include "util/functional.h"
#include "fd/grid.h"
#include "types.h"
#include "delta.h"

namespace ib {
namespace novel {

struct index {
	int index;
	int lower;
	int upper;

	constexpr struct index&
	operator*=(const struct index& idx)
	{
		auto k = [] (const struct index& idx)
		{
			return idx.lower <= idx.index &&
				   idx.index <  idx.upper ?
				   idx.index :  idx.lower-1;
		};
		auto weight = upper - lower;
		index = k(*this) + weight * k(idx);
		lower = weight * idx.lower;
		upper = weight * idx.upper;
		return *this;
	}

	constexpr int value() const { return index; }
	constexpr operator int() const { return value(); }
};

constexpr index
operator*(index l, const index& r)
{
	l *= r;
	return l;
}

template <typename grid_type>
struct indexer {
public:
	using point_type = typename grid_type::point_type;
	using units_type = typename grid_type::units_type;
	using indices_type = typename grid_type::indices_type;
private:
	const grid_type& _grid;

	constexpr decltype(auto)
	components() const
	{
		return _grid.components();
	}

	template <typename point_type, typename indexing_type>
	constexpr int
	build(const point_type& p, indexing_type&& indexing) const
	{
		using namespace util::functional;
		constexpr std::multiplies<void> op = {};

		auto indices = map(indexing, p, components());
		return apply(partial(foldl, op), indices);
	}
public:
	constexpr auto
	decompose(int index) const
	{
		using namespace util::functional;
		auto assign = [] (auto& dst, const auto& src) { dst = src; };
		auto k = [&] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto points = comp.points();
			auto weight = points + solid;
			auto u = (index + solid) % weight - solid;
			index /= weight;
			return u;
		};

		indices_type indices = {0};
		map(assign, indices, map(k, components()));
		return indices;
	}

	constexpr auto
	sort(const units_type& u) const
	{
		auto k = [] (double v, const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto shift = comp.shift();
			auto i = (int) util::math::floor(v - shift);
			return index{i, -solid, comp.points()};
		};
		return build(u, k);
	}

	constexpr auto
	grid(const indices_type& u) const
	{
		auto k = [] (int i, const auto& comp)
		{
			auto j = comp.index(i);
			return index{j, 0, comp.points()};
		};
		return build(u, k);
	}

	constexpr indexer(const grid_type& grid) :
		_grid(grid) {}
};

template <std::size_t dimensions>
struct sweep_info {
	static constexpr auto total_values = 1 << (2 * dimensions);
	static constexpr auto values_per_sweep = 1 << dimensions;
	static constexpr auto sweeps = (total_values + values_per_sweep - 1) / values_per_sweep;
	using container_type = values_container<values_per_sweep>;
};

template <typename grid_type>
struct sweep {
	static constexpr auto dimensions = grid_type::dimensions;
	using info = sweep_info<dimensions>;
	static constexpr auto values_per_sweep = info::values_per_sweep;
	static constexpr auto total_values = info::total_values;
	static constexpr auto sweeps = info::sweeps;
	using container_type = typename info::container_type;
	static constexpr cosine_delta phi = {};

	int count;
	const grid_type& grid;

	static constexpr auto mask(int i, int m, int n) { return (i & (m << n)) >> n; }

	constexpr auto
	values(const delta<dimensions>& dx, double f) const
	{
		container_type values = {0.0};
		if (count >= sweeps)
			return values;

		double weights[dimensions][2];
		for (int i = 0; i < dimensions; ++i) {
			auto base = mask(count, 1, i) - 1;
			auto v = phi(base + dx[i]);
			weights[i][0] = v;
			weights[i][1] = 0.5 - v;
		}

		for (int i = 0; i < values_per_sweep; ++i) {
			double v = f;
			for (int j = 0; j < dimensions; ++j)
				v *= weights[j][mask(i, 1, j)];
			values[i] = v;
		}

		return values;
	}

	constexpr auto
	indices(int index) const
	{
		std::array<int, values_per_sweep> values = {0};
		indexer idx{grid};

		auto indices = idx.decompose(index);
		for (int i = 0; i < values_per_sweep; ++i) {
			auto base = mask(count, 1, i) - 1;
			shift<dimensions> s = {0};
			for (int j = 0; j < dimensions; ++j)
				s[j] = base + 2 * mask(i, 3, j);
			values[i] = idx.grid(indices + s);
		};

		return values;
	}

	constexpr sweep(int count, const grid_type& grid) :
		count(count), grid(grid) {}

};

} // namespace novel
} // namespace ib
