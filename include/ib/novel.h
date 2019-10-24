#pragma once
#include "util/math.h"
#include "util/functional.h"

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

} // namespace novel
} // namespace ib
