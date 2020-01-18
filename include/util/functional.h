#pragma once
#include <functional>
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include "debug.h"

namespace util {

namespace functional {
namespace impl {

template <typename tuple_type>
using plain_tuple_t = std::remove_reference_t<std::remove_cv_t<tuple_type>>;

template <std::size_t i, typename tuple_type>
struct tuple_element : std::tuple_element<i, tuple_type> {};

template <std::size_t i, typename value_type, std::size_t size>
struct tuple_element<i, value_type[size]> {
	static_assert(i < size, "index exceeds array bounds");
	using type = value_type;
};

template <std::size_t i, typename value_type, std::size_t size>
struct tuple_element<i, const value_type[size]> {
	static_assert(i < size, "index exceeds array bounds");
	using type = std::add_const_t<typename tuple_element<i, value_type[size]>::type>;
};

template <std::size_t i, typename value_type, std::size_t size>
struct tuple_element<i, volatile value_type[size]> {
	static_assert(i < size, "index exceeds array bounds");
	using type = std::add_volatile_t<typename tuple_element<i, value_type[size]>::type>;
};

template <std::size_t i, typename value_type, std::size_t size>
struct tuple_element<i, const volatile value_type[size]> {
	static_assert(i < size, "index exceeds array bounds");
	using type = std::add_cv_t<typename tuple_element<i, value_type[size]>::type>;
};

template <std::size_t i, typename int_type, int_type ... n>
struct tuple_element<i, std::integer_sequence<int_type, n...>> {
	static constexpr int_type arr[sizeof...(n)] = {n...};
	static_assert(i < sizeof...(n), "index exceeds array bounds");
	using type = std::integral_constant<int_type, arr[i]>;
};

template <std::size_t i, typename int_type, int_type ... n>
struct tuple_element<i, const std::integer_sequence<int_type, n...>> {
	static constexpr int_type arr[sizeof...(n)] = {n...};
	static_assert(i < sizeof...(n), "index exceeds array bounds");
	using type = std::integral_constant<int_type, arr[i]>;
};

template <std::size_t i, typename int_type, int_type ... n>
struct tuple_element<i, volatile std::integer_sequence<int_type, n...>> {
	static constexpr int_type arr[sizeof...(n)] = {n...};
	static_assert(i < sizeof...(n), "index exceeds array bounds");
	using type = std::integral_constant<int_type, arr[i]>;
};

template <std::size_t i, typename int_type, int_type ... n>
struct tuple_element<i, const volatile std::integer_sequence<int_type, n...>> {
	static constexpr int_type arr[sizeof...(n)] = {n...};
	static_assert(i < sizeof...(n), "index exceeds array bounds");
	using type = std::integral_constant<int_type, arr[i]>;
};

template <std::size_t i, typename tuple_type>
using tuple_element_t = typename tuple_element<i, tuple_type>::type;

template <typename> struct sfinae_pass : std::true_type {};

template <typename tuple_type>
struct tuple_size : std::tuple_size<tuple_type> {};

template <std::size_t i, typename tuple_type>
struct has_get_method {
	template <typename C> static auto test(int) ->
		sfinae_pass<decltype(std::declval<C>().template get<i>())>;
	template <typename C> static auto test(long) -> std::false_type;
	static constexpr auto value = decltype(test<tuple_type>(0))::value;
};

template <typename value_type, std::size_t size>
struct tuple_size<value_type[size]> :
	std::integral_constant<std::size_t, size> {};

template <typename int_type, int_type ... n>
struct tuple_size<std::integer_sequence<int_type, n...>> :
	std::integral_constant<std::size_t, sizeof...(n)> {};

template <typename int_type, int_type ... n>
struct tuple_size<const std::integer_sequence<int_type, n...>> :
	std::integral_constant<std::size_t, sizeof...(n)> {};

template <typename tuple_type>
inline constexpr auto tuple_size_v = tuple_size<tuple_type>::value;

template <std::size_t i, typename tuple_type>
constexpr decltype(auto)
internal_get(tuple_type&& tuple) noexcept
{
	using std::get;
	if constexpr (has_get_method<i, std::remove_reference_t<tuple_type>>::value)
		return tuple.template get<i>();
	else
		/* This call does not necessarily call std::get if
		 * it is not overloaded for the argument; it will use
		 * ADL if a valid std::get is not found. */
		return get<i>(std::forward<tuple_type>(tuple));
}

template <std::size_t i, typename value_type, std::size_t size>
constexpr value_type&
internal_get(value_type (&arr)[size])
{
	return arr[i];
}


template <std::size_t i, typename value_type, std::size_t size>
constexpr const value_type&
internal_get(const value_type (&arr)[size])
{
	return arr[i];
}

template <std::size_t i, typename int_type, int_type ... n>
constexpr decltype(auto)
internal_get(std::integer_sequence<int_type, n...>)
{
	constexpr int_type values[] = {n...};
	return std::integral_constant<int_type, values[i]>{};
}

template <typename first_type, typename second_type, std::size_t ... n, std::size_t ... m>
constexpr decltype(auto)
cat_pair(first_type&& first, second_type&& second, std::index_sequence<n...>, std::index_sequence<m...>)
{
	return std::tuple<decltype(internal_get<n>(std::forward<first_type>(first)))...,
		   decltype(internal_get<m>(std::forward<second_type>(second)))...>{
		internal_get<n>(std::forward<first_type>(first))...,
		internal_get<m>(std::forward<second_type>(second))...};
}

template <typename tuple_type>
constexpr decltype(auto)
tuple_cat(tuple_type&& tuple)
{
	return tuple;
}

template <typename first_type, typename second_type>
constexpr decltype(auto)
tuple_cat(first_type&& first, second_type&& second)
{
	constexpr auto first_size = tuple_size_v<plain_tuple_t<first_type>>;
	constexpr auto second_size = tuple_size_v<plain_tuple_t<second_type>>;
	using first_sequence = std::make_index_sequence<first_size>;
	using second_sequence = std::make_index_sequence<second_size>;
	return cat_pair(std::forward<first_type>(first),
			std::forward<second_type>(second),
			first_sequence{}, second_sequence{});
}

template <typename first_type, typename second_type, typename ... rest_types>
constexpr decltype(auto)
tuple_cat(first_type&& first, second_type&& second, rest_types&& ... rest)
{
	return tuple_cat(tuple_cat(std::forward<first_type>(first), std::forward<second_type>(second)),
			std::forward<rest_types>(rest)...);
}

template <bool ...>
struct all : std::true_type {};

template <bool ... Bs>
struct all<true, Bs...> : std::integral_constant<bool, all<Bs...>::value> {};

template <bool ... Bs>
struct all<false, Bs...> : std::false_type {};

template <bool ... Bs>
struct any { static constexpr bool value = !all<!Bs...>::value; };

template <bool ... Bs> inline constexpr auto any_v = any<Bs...>::value;
template <bool ... Bs> inline constexpr auto all_v = all<Bs...>::value;


template <std::size_t ... Ns, typename Fn, typename Tpl>
constexpr decltype(auto)
apply(std::index_sequence<Ns...>, Fn&& fn, Tpl&& args)
{
	return fn(internal_get<Ns>(std::forward<Tpl>(args))...);
}

struct apply_functor {
	template <typename Fn, typename Tpl>
	constexpr decltype(auto)
	operator()(Fn&& fn, Tpl&& args) const
	{
		using sequence = std::make_index_sequence<std::tuple_size_v<plain_tuple_t<Tpl>>>;
		return apply(sequence{}, std::forward<Fn>(fn), std::forward<Tpl>(args));
	}
};

template <std::size_t N, typename ... Ts>
constexpr decltype(auto)
zip(Ts&& ... ts)
{
	return std::forward_as_tuple(internal_get<N>(std::forward<Ts>(ts))...);
}

struct zip_functor {
private:
	template <std::size_t ... Ns, typename ... Tpls>
	static constexpr decltype(auto)
	call(const std::index_sequence<Ns...>&, Tpls&& ... tpls)
	{
		return std::make_tuple(zip<Ns>(std::forward<Tpls>(tpls)...)...);
	}
public:
	template <typename Tpl, typename ... Tpls>
	constexpr decltype(auto)
	operator()(Tpl&& tpl, Tpls&& ... tpls) const
	{
		using sequence = std::make_index_sequence<tuple_size_v<plain_tuple_t<Tpl>>>;
		return call(sequence(), std::forward<Tpl>(tpl), std::forward<Tpls>(tpls)...);
	}

	constexpr std::tuple<>
	operator()() const
	{
		return std::tuple<>();
	}
};

template <typename ...> struct all_same;
template <typename ... types>
inline constexpr bool all_same_v = all_same<types...>::value;

template <> struct all_same<> : std::true_type {};
template <typename first> struct all_same<first> : std::true_type {};
template <typename first, typename ... types>
struct all_same<first, types...> : all<std::is_same_v<first, types>...> {};

template <bool, typename ...> struct map_tuple_type;
template <typename first, typename ... types>
struct map_tuple_type<true, first, types...> {
	using type = std::array<first, 1 + sizeof...(types)>;
};
template <typename ... types>
struct map_tuple_type<false, types...> {
	using type = std::tuple<types...>;
};
template <bool b> struct map_tuple_type<b> { using type = std::tuple<>; };
template <typename ... types>
using map_tuple = std::conditional_t<any_v<
	std::is_void_v<types>...>, void, std::tuple<types...>>;

template <typename ... arg_types>
constexpr void swallow(const arg_types& ...) {}

template <std::size_t ... Ns, typename Fn, typename Tpl>
constexpr decltype(auto)
map(std::index_sequence<Ns...>, Fn&& fn, Tpl&& tpl)
{
	using tuple_type = map_tuple<decltype(fn(internal_get<Ns>(std::forward<Tpl>(tpl))))...>;
	constexpr bool returns_void = std::is_void_v<tuple_type>;
	if constexpr (!returns_void)
		return tuple_type{fn(internal_get<Ns>(std::forward<Tpl>(tpl)))...};
	else
		return swallow((fn(internal_get<Ns>(std::forward<Tpl>(tpl))), 0)...);
}

template <typename Fn, typename ... Args>
class partial {
private:
	using tuple_type = std::tuple<Args...>;
	Fn fn;
	tuple_type args;
public:
	template <typename ... Ts>
	constexpr decltype(auto)
	operator()(Ts&& ... ts) const
	{
		auto new_args = tuple_cat(args, std::tuple<Ts&&...>(std::forward<Ts>(ts)...));
		using sequence = std::make_index_sequence<sizeof...(Args) + sizeof...(Ts)>;
		return apply(sequence{}, fn, std::move(new_args));
	}

	constexpr partial(Fn&& fn, Args&& ... args) :
		fn(std::forward<Fn>(fn)), args{std::forward<Args>(args)...} {}
};

struct partial_functor {
	template <typename Fn, typename ... Args>
	constexpr decltype(auto)
	operator()(Fn&& fn, Args&& ... args) const
	{
		return partial<Fn, Args...>{std::forward<Fn>(fn), std::forward<Args>(args)...};
	}
};

struct map_functor {
	template <typename Fn, typename Tpl>
	constexpr decltype(auto)
	operator()(Fn&& fn, Tpl&& tpl) const
	{
		using sequence = std::make_index_sequence<tuple_size_v<plain_tuple_t<Tpl>>>;
		return map(sequence(), std::forward<Fn>(fn), std::forward<Tpl>(tpl));
	}

	template <typename Fn, typename ... Tpls>
	constexpr decltype(auto)
	operator()(Fn&& fn, Tpls&& ... tpls) const
	{
		constexpr apply_functor apply;
		constexpr zip_functor zip;
		constexpr partial_functor partial;
		return operator()(partial(apply, std::forward<Fn>(fn)),
				zip(std::forward<Tpls>(tpls)...));
	}
};

struct bind_functor {
	template <typename Fn, typename ... Args>
	constexpr decltype(auto)
	operator()(Fn&& fn, Args&& ... args) const
	{
		return std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
	}
};

template <typename Fn, typename T0>
constexpr decltype(auto)
foldl(Fn&& fn, T0&& t0)
{
	return std::forward<T0>(t0);
}

template <typename Fn, typename T0, typename T1>
constexpr decltype(auto)
foldl(Fn&& fn, T0&& t0, T1&& t1)
{
	return fn(std::forward<T0>(t0), std::forward<T1>(t1));
}

template <typename Fn, typename T0, typename T1, typename ... Ts>
constexpr decltype(auto)
foldl(Fn&& fn, T0&& t0, T1&& t1, Ts&& ... ts)
{
	auto&& tn = fn(std::forward<T0>(t0), std::forward<T1>(t1));
	return foldl(std::forward<Fn>(fn), std::move(tn), std::forward<Ts>(ts)...);
}

struct foldl_functor {
	template <typename Fn, typename ... Ts>
	constexpr decltype(auto)
	operator()(Fn&& fn, Ts&& ... ts) const
	{
		return foldl(std::forward<Fn>(fn), std::forward<Ts>(ts)...);
	}
};

template <typename Fn, typename T0>
constexpr decltype(auto)
foldr(Fn&& fn, T0&& t0)
{
	return std::forward<T0>(t0);
}

template <typename Fn, typename T0, typename T1>
constexpr decltype(auto)
foldr(Fn&& fn, T0&& t0, T1&& t1)
{
	return fn(std::forward<T0>(t0), std::forward<T1>(t1));
}

template <typename Fn, typename T0, typename ... Ts>
constexpr decltype(auto)
foldr(Fn&& fn, T0&& t0, Ts&& ... ts)
{
	return fn(std::forward<T0>(t0), foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...));
}

struct foldr_functor {
	template <typename Fn, typename ... Ts>
	constexpr decltype(auto)
	operator()(Fn&& fn, Ts&& ... ts) const
	{
		return foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...);
	}
};

template <std::size_t ... Ns, typename Tpl>
constexpr decltype(auto)
reverse(const std::index_sequence<Ns...>&, Tpl&& tpl)
{
	using tuple_type = plain_tuple_t<Tpl>;
	constexpr auto size = tuple_size_v<tuple_type>;
	return std::forward_as_tuple(internal_get<size-1-Ns>(std::forward<Tpl>(tpl))...);
}

struct reverse_functor {
	template <typename Tpl>
	constexpr decltype(auto)
	operator()(Tpl&& tpl) const
	{
		using sequence = std::make_index_sequence<tuple_size_v<plain_tuple_t<Tpl>>>;
		return reverse(sequence{}, std::forward<Tpl>(tpl));
	}
};

template <int iterations, int init=0, int stride=1>
struct iterate_functor {
	template <typename Fn>
	constexpr decltype(auto)
	operator()(Fn&& fn) const
	{
		auto k = [fn=std::forward<Fn>(fn)] (auto m)
		{
			constexpr auto i = init + stride * m;
			return fn(std::integral_constant<int, i>{});
		};
		return map(k, std::make_integer_sequence<int, iterations>{});
	}
};

} // namespace impl

inline constexpr impl::zip_functor     zip;
inline constexpr impl::apply_functor   apply;
inline constexpr impl::map_functor     map;
inline constexpr impl::bind_functor    bind;
inline constexpr impl::partial_functor partial;
inline constexpr impl::foldl_functor   foldl;
inline constexpr impl::foldr_functor   foldr;
inline constexpr impl::reverse_functor reverse;
template <int iterations, int init=0, int stride=1>
inline constexpr impl::iterate_functor<iterations, init, stride> iterate;

} // namespace functional
} // namespace util
