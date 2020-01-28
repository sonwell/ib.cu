#pragma once
#include <iostream>
#include "debug.h"
#include "functional.h"

namespace util {
namespace iterators {

template <typename int_type>
struct counter {
private:
	int_type from, to;

	struct counting_iterator {
	private:
		int_type i;
	public:
		using value_type = int_type;
		using iterator_category = std::input_iterator_tag;

		constexpr auto& operator++() { ++i; return *this; }
		constexpr auto  operator++(int) { auto cp = *this; ++i; return cp; }
		constexpr int_type operator*() const { return i; }
		constexpr bool operator==(const counting_iterator& o) const { return i == o.i; }
		constexpr bool operator!=(const counting_iterator& o) const { return i != o.i; }
		constexpr counting_iterator(int_type i) : i(i) {}
	};
public:
	using iterator = counting_iterator;
	using const_iterator = counting_iterator;

	constexpr auto begin() { return iterator{from}; }
	constexpr auto end() { return iterator{to}; }
	constexpr auto begin() const { return const_iterator{from}; }
	constexpr auto end() const { return const_iterator{to}; }

	constexpr counter(int_type from, int_type to) :
		from(from), to(to) {}
};

template <typename ... iterable_types>
struct zip {
private:
	std::tuple<iterable_types...> iterables;

	template <typename ... iterator_types>
	struct zip_iterator {
	private:
		using container = std::tuple<iterator_types...>;
		container its;

		template <typename k_type>
		constexpr auto
		any(const zip_iterator& it, k_type&& k) const
		{
			using namespace util::functional;
			auto r = [&] (auto&& ... v) { return (apply(k, v) || ...); };
			return apply(r, zip(its, it.its));
		}

		template <typename k_type>
		constexpr auto
		all(const zip_iterator& it, k_type&& k) const
		{
			using namespace util::functional;
			auto r = [&] (auto&& ... v) { return (apply(k, v) && ...); };
			return apply(r, util::functional::zip(its, it.its));
		}

		constexpr void
		advance()
		{
			using namespace util::functional;
			map([] (auto& it) { ++it; }, its);
		}
	public:
		constexpr decltype(auto)
		operator*() const
		{
			using namespace util::functional;
			auto deref = [] (auto&& t) { return *t; };
			return map(deref, its);
		}

		constexpr decltype(auto) operator++() { advance(); return *this; }
		constexpr decltype(auto) operator++(int) { auto cp = *this; advance(); return cp; }

		constexpr decltype(auto)
		operator==(const zip_iterator& o) const
		{
			return any(o, [] (auto& l, auto& r) { return l == r; });
		}

		constexpr decltype(auto)
		operator!=(const zip_iterator& o) const
		{
			return all(o, [] (auto& l, auto& r) { return l != r; });
		}

		constexpr zip_iterator(iterator_types ... its) : its{its...} {}

		using value_type = decltype(*std::declval<zip_iterator>());
	};
public:
	using iterator = zip_iterator<decltype(std::declval<iterable_types>().begin())...>;
	using const_iterator = zip_iterator<decltype(std::declval<const iterable_types>().begin())...>;

	constexpr auto
	begin()
	{
		using namespace util::functional;
		constexpr auto v = [] (auto ... v) { return iterator{std::move(v)...}; };
		return apply(v, map([] (auto& it) { return it.begin(); }, iterables));
	}

	constexpr auto
	end()
	{
		using namespace util::functional;
		constexpr auto v = [] (auto ... v) { return iterator{std::move(v)...}; };
		return apply(v, map([] (auto& it) { return it.end(); }, iterables));
	}

	constexpr auto
	begin() const
	{
		using namespace util::functional;
		constexpr auto v = [] (auto ... v) { return const_iterator{std::move(v)...}; };
		return apply(v, map([] (auto& it) { return it.begin(); }, iterables));
	}

	constexpr auto
	end() const
	{
		using namespace util::functional;
		constexpr auto v = [] (auto ... v) { return const_iterator{std::move(v)...}; };
		return apply(v, map([] (auto& it) { return it.end(); }, iterables));
	}

	constexpr zip(iterable_types ... iterables) :
		iterables{std::move(iterables)...} {}
};

template <typename iterable_type, typename transform_type>
struct transform {
private:
	iterable_type iterable;
	transform_type fn;

	template <typename iterator_type>
	struct transform_iterator {
	private:
		iterator_type it;
		transform_type fn;
	public:
		constexpr auto& operator++() { ++it; return *this; }
		constexpr auto  operator++(int) { auto cp = *this; ++it; return cp; }
		constexpr decltype(auto) operator*() const { return fn(*it); }
		constexpr bool operator==(const transform_iterator& o) const { return it == o.it; }
		constexpr bool operator!=(const transform_iterator& o) const { return it != o.it; }

		constexpr transform_iterator(iterator_type it, transform_type fn) :
			it(it), fn(fn) {}
	};
public:
	using const_iterator = transform_iterator<
		typename iterable_type::const_iterator>;
	using iterator = transform_iterator<
		typename iterable_type::iterator>;

	constexpr auto begin() { return const_iterator{iterable.begin(), fn}; }
	constexpr auto end() { return const_iterator{iterable.end(), fn}; }
	constexpr auto begin() const { return iterator{iterable.begin(), fn}; }
	constexpr auto end() const { return iterator{iterable.end(), fn}; }

	constexpr transform(iterable_type iterable, transform_type fn) :
		iterable(std::move(iterable)), fn(std::move(fn)) {}
};

template <typename iterable_type>
struct enumerate : zip<counter<int>, iterable_type> {
private:
	using base = zip<counter<int>, iterable_type>;
public:
	constexpr enumerate(iterable_type iterable, int start = 0) :
		base(counter<int>(start, start-1), std::move(iterable)) {}
};

} // namespace iterators
} // namespace util
