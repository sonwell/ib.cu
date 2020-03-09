#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "types.h"

namespace ib {
namespace indexing {

struct clamped {
	static constexpr auto
	clamp(int index, int weight)
	{
		return index < 0 ? 0 :
			index >= weight ? weight - 1 : index;
	}

	int index, weight;

	constexpr operator int() const { return index; }

	constexpr clamped(int index, int lower, int upper) :
		index(clamp(index-lower, upper-lower)),
		weight(upper-lower) {}
};

constexpr clamped
combine(const clamped& l, const clamped& r)
{
	auto index = l.index + l.weight * r.index;
	auto weight = l.weight * r.weight;
	return {index, 0, weight};
}

struct errors_negative {
private:
	static constexpr
	auto clamp(int index, int weight)
	{
		return 0 <= index && index < weight ? index : -1;
	}
public:
	int index, weight;

	explicit constexpr operator int() const { return index; }

	constexpr errors_negative(int index, int lower, int upper) :
		index(clamp(index-lower, upper-lower)),
		weight(upper-lower) {}

	friend constexpr errors_negative
		combine(const errors_negative&, const errors_negative&);
};

constexpr errors_negative
combine(const errors_negative& l, const errors_negative& r)
{
	auto index = l.index < 0 ? -1 :
		l.index + l.weight * r.index;
	auto weight = l.weight * r.weight;
	return {index, 0, weight};
}

template <typename k_type, typename point_type, typename sorter_type>
constexpr auto
reduce(k_type&& k, point_type&& p, sorter_type&& sorter)
{
	using namespace util::functional;
	auto op = [&] (auto l, auto&& r)
	{
		return combine(std::move(l),
				apply(k, std::forward<decltype(r)>(r)));
	};
	auto r = [&] (auto&& f, auto&& ... r)
	{
		return foldl(op,
				apply(k, std::forward<decltype(f)>(f)),
				std::forward<decltype(r)>(r)...);
	};
	return apply(r, zip(p, components(sorter)));
}

template <typename grid_type>
struct sorter {
public:
	static constexpr auto dimensions = grid_type::dimensions;
protected:
	using point = ib::point<dimensions>;

	template <typename k_type>
	constexpr auto
	decompose(k_type&& k) const
	{
		using namespace util::functional;
		using indices = ib::indices<dimensions>;
		constexpr auto v = [] (auto ... v) { return indices{v...}; };
		return apply(v, map(k, components()));
	}
public:
	using grid_index_type = errors_negative;
	using sort_index_type = clamped;

	constexpr auto
	decompose(int i) const
	{
		auto k = [&] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto [index, weight] = clamped{0, -solid, comp.points()};
			auto j = i % weight - index;
			i /= weight;
			return j;
		};
		return decompose(k);
	}

	constexpr sorter(const grid_type& g) : grid(g) {}
	friend constexpr auto components(const sorter& s) { return s.components(); }
private:
	constexpr auto components() const { return grid.components(); }
	grid_type grid;
};

} // namespace indexing
} // namespace ib
