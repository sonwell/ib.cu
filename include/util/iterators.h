#pragma once
#include "util/functional.h"

namespace util {

template <typename iter_type>
struct enumerate {
private:
	using iterator_type = typename iter_type::iterator;
	int index;
	iter_type iter;
public:
	struct iterator {
	private:
		int index;
		iterator_type it;

		constexpr void advance() { ++it, ++index; }
	public:
		constexpr decltype(auto) operator*() const { return std::pair<int, decltype(*it)>{index, *it}; }
		constexpr decltype(auto) operator++() { advance(); return *this; }
		constexpr decltype(auto) operator++(int) { iterator cp = *this; advance(); return cp; }
		constexpr decltype(auto) operator==(const iterator& o) const { return it == o.it; }
		constexpr decltype(auto) operator!=(const iterator& o) const { return it != o.it; }
		constexpr iterator(int index, iterator_type it) : index(index), it(it) {}
	};

	constexpr enumerate(iter_type iter, int start=0) :
		index(start), iter(std::move(iter)) {}

	constexpr auto begin() const { return iterator{index, iter.begin()}; }
	constexpr auto end() const { return iterator{index, iter.end()}; }
};

template <typename ... iter_types>
struct zip {
private:
	template <typename T> using iterator_type = typename T::iterator;
	std::tuple<iter_types...> iters;

	template <typename k_type>
	constexpr auto
	construct(k_type&& k) const
	{
		using namespace util::functional;
		auto ctor = [&] (auto& ... it) { return iterator{k(it)...}; };
		return apply(ctor, iters);
	}
public:
	struct iterator {
	private:
		using container = std::tuple<iterator_type<iter_types>...>;
		container its;

		template <typename k_type>
		constexpr auto
		any(const iterator& it, k_type&& k) const
		{
			using namespace util::functional;
			auto r = [&] (auto&& ... v) { return (apply(k, v) || ...); };
			return apply(r, zip(its, it.its));
		}

		template <typename k_type>
		constexpr auto
		all(const iterator& it, k_type&& k) const
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
		operator==(const iterator& o) const
		{
			return any(o, [] (auto& l, auto& r) { return l == r; });
		}

		constexpr decltype(auto)
		operator!=(const iterator& o) const
		{
			return all(o, [] (auto& l, auto& r) { return l != r; });
		}

		constexpr iterator(iterator_type<iter_types> ... its) : its{its...} {}
	};

	constexpr auto
	begin() const
	{
		constexpr auto k = [] (auto& it) { return it.begin(); };
		return construct(k);
	}

	constexpr auto
	end() const
	{
		constexpr auto k = [] (auto& it) { return it.end(); };
		return construct(k);
	}

	constexpr zip(iter_types ... iters) : iters{std::move(iters)...} {}
};

} // namespace util
