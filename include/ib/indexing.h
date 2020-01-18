#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "types.h"

namespace ib {
namespace indexing {

struct unclamped {
private:
	int index, lower = 0, upper = 1;
public:
	explicit constexpr operator int() const { return index; }

	constexpr unclamped(int index, int lower, int upper) :
		index(index), lower(lower), upper(upper) {}

	friend constexpr unclamped
		combine(const unclamped&, const unclamped&);
};

constexpr unclamped
combine(const unclamped& l, const unclamped& r)
{
	auto weight = l.upper - l.lower;
	auto index = l.index + weight * r.index;
	auto lower = weight * r.lower;
	auto upper = weight * r.upper;
	return unclamped{index, lower, upper};
}

struct errors_negative {
private:
	static constexpr
	auto clamp(int index, int lower, int upper)
	{
		return lower <= index && index < upper ? index : lower-1;
	}

	int index, lower = 0, upper = 1;
public:
	explicit constexpr operator int() const { return index; }

	constexpr errors_negative(int index, int lower, int upper) :
		index(clamp(index, lower, upper)),
		lower(lower), upper(upper) {}

	friend constexpr errors_negative
		combine(const errors_negative&, const errors_negative&);
};

constexpr errors_negative
combine(const errors_negative& l, const errors_negative& r)
{
	auto weight = l.upper - l.lower;
	auto index = l.index + weight * r.index;
	auto lower = weight * r.lower;
	auto upper = weight * r.upper;
	return errors_negative{index, lower, upper};
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
protected:
	static constexpr auto dimensions = grid_type::dimensions;
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
	using sort_index_type = unclamped;

	constexpr auto
	decompose(int i) const
	{
		auto k = [&] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto weight = comp.points() + solid;
			auto j = (i + solid) % weight - solid;
			i /= weight;
			return j;
		};
		return decompose(k);
	}

	constexpr auto
	difference(const point& p) const
	{
		using namespace util::functional;
		using difference = ib::difference<dimensions>;
		constexpr auto v = [] (auto ... v) { return difference{v...}; };
		auto k = [] (double x, const auto& comp)
		{
			auto shift = comp.shift();
			auto diff = comp.units(x) - shift;
			return ((int) diff) - diff;
		};
		return apply(v, map(k, p, components()));
	}

	constexpr sorter(const grid_type& g) : grid(g) {}
	friend constexpr auto components(const sorter& s) { return s.components(); }
private:
	constexpr auto components() const { return grid.components(); }
	grid_type grid;
};

} // namespace indexing

template <typename sorter_type>
struct indexer {
public:
	static constexpr auto dimensions = std::tuple_size_v<
		decltype(components(std::declval<sorter_type>()))>;
private:
	using indices = ib::indices<dimensions>;
	using point = ib::point<dimensions>;
public:
	constexpr auto decompose(int index) const { return s.decompose(index); }
	constexpr auto difference(const point& p) const { return s.difference(p); }

	constexpr auto
	sort(const point& p) const
	{
		using index = typename sorter_type::sort_index_type;
		constexpr auto k = [] (double x, const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto shift = comp.shift();
			auto i = (int) (comp.units(x) - shift);
			return index{i, -solid, comp.points()};
		};
		return (int) indexing::reduce(k, p, s);
	}

	constexpr auto
	grid(const indices& p) const
	{
		using index = typename sorter_type::grid_index_type;
		constexpr auto k = [&] (int i, const auto& comp)
		{
			auto j = comp.index(i);
			return index{j, 0, comp.points()};
		};
		return (int) indexing::reduce(k, p, s);
	}

	constexpr indexer(const sorter_type& s) : s(s) {}
private:
	sorter_type s;
};

} // namespace ib
