#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "types.h"
#include "indexing.h"
#include "delta.h"

namespace ib {
namespace detail {

template <std::size_t dimensions>
using shift = util::wrapper<struct shift_tag, std::array<int, dimensions>>;

template <std::size_t dimensions>
constexpr auto
operator+(indices<dimensions> l, shift<dimensions> r)
{
	using namespace util::functional;
	map([] (int& l, const int& r) { l += r; }, l, r);
	return l;
}

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
struct weights {
private:
	using traits = ib::delta::traits<delta_type>;
	static constexpr auto dimensions = indexer_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto calls = traits::calls;
	static constexpr auto pattern = traits::pattern(dimensions);
	static constexpr auto values = detail::cpow(meshwidths, dimensions);
	using vec = delta::detail::vec<calls>;
	using point_type = std::array<double, dimensions>;

	template <typename f_type>
	static constexpr auto
	rules(int i, f_type&& f)
	{
		return [=, f=std::forward<f_type>(f)] (auto&& ... args) mutable
		{
			constexpr auto rules = traits::rules;
			auto j = i % meshwidths;
			i /= meshwidths;
			const auto& rule = rules[j];
			return f(rule, std::forward<decltype(args)>(args)...);
		};
	};

	static constexpr auto
	flags(int offset, int count)
	{
		using namespace util::functional;
		using flags_type = std::array<bool, calls>;
		constexpr auto a = [] (bool& l, auto&& r) { l |= !!r; };
		constexpr auto k = [=] (auto& r, flags_type& l) { map(a, l, r.r.weights); };

		std::array<flags_type, dimensions> flags = {};
		for (int i = 0; i < count; ++i)
			map(rules(pattern[offset + i], k), flags);
		return flags;
	}

	static constexpr auto
	evaluate(int offset, int count, const delta_type& phi, const point_type& r)
	{
		using namespace util::functional;
		auto fl = flags(offset, count);
		auto v = [] (auto ... v) { return vec{std::move(v)...}; };
		auto e = [&] (double r, const bool& flag, const int& offset)
		{
			return flag ? phi(r + offset) : 0.0;
		};
		auto k = [&] (double r, const auto& f)
		{
			constexpr auto offsets = traits::offsets;
			return apply(v, map(partial(e, r), f, offsets));
		};
		return map(k, r, fl);
	}

	using phi_type = decltype(evaluate(0, 0,
				std::declval<delta_type>(),
				std::declval<point_type>()));

	constexpr weights(int offset, int remaining, const delta_type& phi,
			const point_type& r) :
		offset(offset), remaining(remaining), r(r),
		phi(evaluate(offset, remaining, phi, r)) {}
public:
	struct iterator {
		int index;
		const weights& data;

		constexpr auto compute(int i) const { return data[i]; }
		constexpr bool operator==(const iterator& o) const { return index == o.index; }
		constexpr bool operator!=(const iterator& o) const { return index != o.index; }
		constexpr auto operator[](int i) const { return compute(i); }
		constexpr auto operator*() const { return compute(index); }
		constexpr auto& operator++() { ++index; return *this; }
		constexpr auto operator++(int) { auto cp = *this; ++index; return cp; }
		constexpr auto& operator+=(int n) { index += n; return *this; }
	};

	constexpr auto
	operator[](int i) const
	{
		using namespace util::functional;
		constexpr auto p = [] (auto ... v) { return (v * ... * 1); };
		constexpr auto k =
			[] (auto& f, double r, const vec& v) { return f.p(r) + f.r * v; };
		auto f = rules(pattern[offset + i], k);
		return apply(p, map(f, r, phi));
	}

	constexpr iterator begin() const { return iterator{0, *this}; }
	constexpr iterator end() const { return iterator{remaining, *this}; }

	constexpr weights(int offset, int remaining, const delta_type& phi,
			const indexer_type& idx, const point_type& p) :
		weights(offset, remaining, phi, idx.difference(p)) {}
private:
	int offset, remaining;
	const point_type& r;
	phi_type phi;
};

template <typename delta_type, typename indexer_type>
struct ordering {
private:
	using traits = delta::traits<delta_type>;
	static constexpr auto dimensions = indexer_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto values = detail::cpow(meshwidths, dimensions);
	static constexpr auto pattern = traits::pattern(dimensions);
	using indices_type = decltype(std::declval<indexer_type>().decompose(0));

	template <typename f_type>
	static constexpr auto
	rules(int i, f_type&& f)
	{
		return [=, f=std::forward<f_type>(f)] (auto&& ... args) mutable
		{
			auto j = i % meshwidths;
			i /= meshwidths;
			return f(j, std::forward<decltype(args)>(args)...);
		};
	}
public:
	struct iterator {
		int index;
		const ordering& data;

		constexpr auto compute(int i) const { return data[i]; }
		constexpr bool operator==(const iterator& o) const { return index == o.index; }
		constexpr bool operator!=(const iterator& o) const { return index != o.index; }
		constexpr auto operator[](int i) const { return compute(i); }
		constexpr auto operator*() const { return compute(index); }
		constexpr auto& operator++() { ++index; return *this; }
	};

	constexpr auto
	operator[](int i) const
	{
		using namespace util::functional;
		constexpr auto delta = (meshwidths - 1) >> 1;
		auto j = pattern[offset + i];
		detail::shift<dimensions> s = {0};
		auto k = [] (int i, int& j) { j = i - delta; };
		map(rules(j, k), s);
		return idx.grid(indices + s);
	}

	constexpr iterator begin() const { return iterator{0, *this}; }
	constexpr iterator end() const { return iterator{remaining, *this}; }

	constexpr ordering(int offset, int remaining, const delta_type&,
			const indexer_type& idx, int sort) :
		offset(offset), remaining(remaining),
		indices(idx.decompose(sort)), idx(idx) {}
private:
	int offset, remaining;
	indices_type indices;
	const indexer_type& idx;
};

template <typename delta_type, typename indexer_type>
struct sweep {
private:
	using traits = delta::traits<delta_type>;
	static constexpr auto dimensions = indexer_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto total_values = detail::cpow(meshwidths, dimensions);
public:
	constexpr auto
	size() const
	{
		return offset + count >= total_values ?
			total_values - offset : count;
	}

	constexpr auto
	index(const point<dimensions>& p) const
	{
		return idx.sort(p);
	}

	constexpr auto
	values(const std::array<double, dimensions>& r) const
	{
		return weights{offset, size(), phi, idx, r};
	}

	constexpr auto
	indices(int sort) const
	{
		return ordering{offset, size(), phi, idx, sort};
	}

	constexpr sweep(int index, int count, delta_type phi, indexer_type idx) :
		offset(index * count), count(count), phi(phi), idx(idx) {}
private:
	int offset, count;
	delta_type phi;
	indexer_type idx;
};

} // namespace ib
