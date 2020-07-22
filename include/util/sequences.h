#pragma once
#include <utility>

namespace util {

template <typename int_type, int_type ... ns>
using sequence = std::integer_sequence<int_type, ns...>;

template <typename int_type, std::size_t n>
using make_sequence = std::make_integer_sequence<int_type, n>;

template <typename int_type, int_type ... ns>
constexpr auto
concat(sequence<int_type, ns...> s)
{
	return s;
}

template <typename int_type, int_type ... ns, int_type ... ms>
constexpr auto
concat(sequence<int_type, ns...>, sequence<int_type, ms...>)
{
	return sequence<int_type, ns..., ms...>();
}

template <typename int_type, int_type ... ns, typename ... arg_types>
constexpr auto
concat(sequence<int_type, ns...> s, arg_types ... args)
{
	return concat(s, concat(args...));
}

template <typename int_type>
constexpr auto
merge(sequence<int_type>,
		sequence<int_type>)
{
	return sequence<int_type>();
}

template <typename int_type, int_type ... ns>
constexpr auto
merge(sequence<int_type, ns...> seq,
		sequence<int_type>)
{
	return seq;
}

template <typename int_type, int_type ... ms>
constexpr auto
merge(sequence<int_type>,
		sequence<int_type, ms...> seq)
{
	return seq;
}

template <typename int_type, int_type n, int_type ... ns,
		 int_type m, int_type ... ms>
constexpr auto
merge(sequence<int_type, n, ns...>,
		sequence<int_type, m, ms...>)
{
	if constexpr(n < m)
		return concat(sequence<int_type, n>(),
				merge(sequence<int_type, ns...>(),
					sequence<int_type, m, ms...>()));
	else
		return concat(sequence<int_type, m>(),
				merge(sequence<int_type, n, ns...>(),
					sequence<int_type, ms...>()));
}

template <typename int_type>
constexpr auto
partition(sequence<int_type> seq)
{
	return std::make_pair(seq, seq);
}

template <typename int_type, int_type n>
constexpr auto
partition(sequence<int_type, n> seq)
{
	return std::make_pair(seq, sequence<int_type>());
}

template <typename int_type, int_type n, int_type m, int_type ... ns>
constexpr auto
partition(sequence<int_type, n, m, ns...>)
{
	auto [l, r] = partition(sequence<int_type, ns...>());
	return std::make_pair(
		concat(sequence<int_type, n>(), l),
		concat(sequence<int_type, m>(), r)
	);
}

template <typename int_type>
constexpr auto
sort(sequence<int_type> seq)
{
	return seq;
}

template <typename int_type, int_type n>
constexpr auto
sort(sequence<int_type, n> seq)
{
	return seq;
}

template <typename int_type, int_type ... ns>
constexpr auto
sort(sequence<int_type, ns...> seq)
{
	auto [l, r] = partition(seq);
	return merge(sort(l), sort(r));
}

template <typename int_type>
constexpr auto
unique(sequence<int_type> seq)
{
	return seq;
}

template <typename int_type, int_type n>
constexpr auto
unique(sequence<int_type, n> seq)
{
	return seq;
}

template <typename int_type, int_type n, int_type m, int_type ... ns>
constexpr auto
unique(sequence<int_type, n, m, ns...>)
{
	auto tail = unique(sequence<int_type, m, ns...>());
	if constexpr(n != m)
		return concat(sequence<int_type, n>(), tail);
	else
		return tail;
}

template <typename int_type>
constexpr auto
reverse(sequence<int_type> seq)
{
	return seq;
}

template <typename int_type, int_type n, int_type ... ns>
constexpr auto
reverse(sequence<int_type, n, ns...>)
{
	return concat(reverse(sequence<int_type, ns...>()),
			sequence<int_type, n>());
}


template <typename int_type, int_type n, int_type ... ns>
constexpr auto
pop(sequence<int_type, n, ns...>)
{
	return sequence<int_type, ns...>();
}

template <typename int_type, int_type ... ns, int_type ... ms>
constexpr auto
prod(sequence<int_type, ns...>,
		sequence<int_type, ms...>)
{
	return sequence<int_type, (ns * ms)...>();
}

template <std::size_t n, typename int_type, int_type ... ns>
constexpr auto
get(sequence<int_type, ns...>) {
	constexpr int_type values[] = {ns...};
	return values[n];
}

template <std::size_t n, typename value_type, value_type ... ns,
		 typename index_type, index_type ... ms>
constexpr auto
embed(sequence<value_type, ns...> v, sequence<index_type, ms...> i)
{
	static_assert(sizeof...(ns) == sizeof...(ms));
	constexpr auto size = sizeof...(ns);
	constexpr value_type values[] = {ns...};
	constexpr index_type indices[] = {ms...};
	constexpr bool recurse = size > 0 && !indices[0];
	if constexpr (!n)
		return sequence<value_type>();
	else if constexpr (recurse)
		return concat(sequence<value_type, values[0]>{},
				embed<n-1>(pop(v), pop(sequence<index_type, (ms-1)...>{})));
	else
		return concat(sequence<value_type, value_type{}>{},
				embed<n-1>(v, sequence<index_type, (ms-1)...>{}));
}

template <std::size_t n, typename value_type, value_type ... ns>
constexpr auto
pad_end(sequence<value_type, ns...> v)
{
	return embed<n>(v, make_sequence<std::size_t, sizeof...(ns)>{});
}

} // namespace util
