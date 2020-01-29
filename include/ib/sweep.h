#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "util/iterators.h"
#include "util/ranges.h"
#include "types.h"
#include "indexing.h"
#include "delta.h"

namespace ib {
namespace detail {

constexpr auto
cpow(int base, int exp)
{
	if (base == 0) return base;
	if (exp == 0) return base/base;
	if (exp == 1) return base;
	auto r = cpow(base, exp>>1);
	return cpow(base, exp&1) * r * r;
}

} // namespace detail

template <typename delta_type, typename indexer_type>
struct sweep {
private:
	using traits = delta::traits<delta_type>;
	static constexpr auto dimensions = indexer_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto total_values = detail::cpow(meshwidths, dimensions);
	static constexpr typename traits::template pattern<dimensions> pattern;
	using shift_type = shift<dimensions>;
	using point = ib::point<dimensions>;
public:
	constexpr auto decompose(int index) const { return idx.decompose(index); }

	constexpr auto
	difference(const point& p) const
	{
		using namespace util::functional;
		using difference = ib::difference<dimensions>;
		constexpr auto v = [] (auto ... v) { return difference{v...}; };
		auto k = [] (double x, const auto& comp)
		{
			constexpr auto rounding = 0.5 * (meshwidths & 1);
			auto shift = comp.shift();
			auto diff = comp.units(x) - shift;
			return ((int) (diff + rounding)) - diff;
		};
		return apply(v, map(k, p, components(idx)));
	}

	constexpr auto
	sort(const point& p) const
	{
		using index = typename indexer_type::sort_index_type;
		constexpr auto k = [] (double x, const auto& comp)
		{
			constexpr auto rounding = 0.5 * (meshwidths & 1);
			auto solid = comp.solid_boundary;
			auto shift = comp.shift();
			auto i = (int) (comp.units(x) - shift + rounding);
			return index{i, -solid, comp.points()};
		};
		return (int) indexing::reduce(k, p, idx);
	}

	constexpr auto
	grid(const indices<dimensions>& p) const
	{
		using index = typename indexer_type::grid_index_type;
		constexpr auto k = [] (int i, const auto& comp)
		{
			auto j = comp.index(i);
			return index{j, 0, comp.points()};
		};
		return (int) indexing::reduce(k, p, idx);
	}

	constexpr auto
	size() const
	{
		return offset + count >= total_values ?
			total_values - offset : count;
	}

	constexpr auto
	values(const point& p) const
	{
		using util::iterators::counter;
		using util::ranges::transform;
		using namespace util::functional;
		auto k = [&, d=difference(p)] (int i)
		{
			auto s = pattern(offset, i);
			auto p = [] (auto ... v) { return (v * ... * 1); };
			auto k = [&] (double r, int s) { return phi(r + s); };
			return apply(p, map(k, d, std::move(s)));
		};
		return counter{0, size()} | transform(k);
	}

	constexpr auto
	indices(int sort) const
	{
		using util::iterators::counter;
		using util::ranges::transform;
		auto k = [&, dec=decompose(sort)] (int i)
		{
			auto s = pattern(offset, i);
			return grid(dec + s);
		};
		return counter{0, size()} | transform(k);
	}

	constexpr sweep(int index, int count,
			const delta_type& phi, const indexer_type& idx) :
		offset(index * count), count(count), phi(phi), idx(idx) {}
private:
	int offset, count;
	delta_type phi;
	indexer_type idx;
};

} // namespace ib
