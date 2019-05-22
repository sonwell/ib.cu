#pragma once
#include <functional>
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include "sequences.h"

namespace util {

template <typename Tpl>
using sequence_matching_tuple = make_sequence<std::size_t,
	  std::tuple_size_v<std::decay_t<Tpl>>>;

namespace functional {
namespace impl {

template <bool ...>
struct all { static constexpr bool value = true; };

template <bool ... Bs>
struct all<true, Bs...> { static constexpr bool value = all<Bs...>::value; };

template <bool ... Bs>
struct all<false, Bs...> { static constexpr bool value = false; };

template <bool ... Bs>
struct any { static constexpr bool value = !all<!Bs...>::value; };

template <bool ... Bs> inline constexpr auto any_v = any<Bs...>::value;
template <bool ... Bs> inline constexpr auto all_v = all<Bs...>::value;


template <typename Fn, typename ... Args>
inline constexpr bool is_invocable = std::is_invocable_v<Fn, Args...>;
template <typename Fn, typename ... Args>
using invoke_result = std::invoke_result_t<Fn, Args...>;

template <typename Fn, typename Tpl, std::size_t ... Ns>
constexpr auto
apply(const std::index_sequence<Ns...>&, Fn&& fn, Tpl&& args) ->
	invoke_result<Fn, decltype(std::get<Ns>(std::forward<Tpl>(args)))...>
{
	return fn(std::get<Ns>(std::forward<Tpl>(args))...);
}

struct apply_functor {
	template <typename Fn, typename Tpl>
	constexpr auto
	operator()(Fn&& fn, Tpl&& args) const
	{
		using sequence = sequence_matching_tuple<Tpl>;
		return apply(sequence(), std::forward<Fn>(fn), std::forward<Tpl>(args));
	}
};

template <std::size_t N, typename ... Ts>
constexpr auto
zip(Ts&& ... ts)
{
	return std::forward_as_tuple(std::get<N>(std::forward<Ts>(ts))...);
}

struct zip_functor {
private:
	template <std::size_t ... Ns, typename ... Tpls>
	static constexpr auto
	call(const std::index_sequence<Ns...>&, Tpls&& ... tpls)
	{
		return std::make_tuple(zip<Ns>(std::forward<Tpls>(tpls)...)...);
	}
public:
	template <typename Tpl, typename ... Tpls>
	constexpr auto
	operator()(Tpl&& tpl, Tpls&& ... tpls) const
	{
		using sequence = sequence_matching_tuple<Tpl>;
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
using map_tuple = typename map_tuple_type<all_same_v<types...>, types...>::type;

template <typename ... arg_types>
constexpr void swallow(const arg_types& ...) {}

template <std::size_t ... Ns, typename Fn, typename Tpl>
constexpr auto
map(const std::index_sequence<Ns...>&, Fn&& fn, Tpl&& tpl)
{
	using arg_tuple = std::decay_t<Tpl>;
	constexpr bool returns_void = any_v<std::is_void_v<
		invoke_result<Fn, std::tuple_element_t<Ns, arg_tuple>>>...>;
	if constexpr (!returns_void) {
		using tuple_type = map_tuple<invoke_result<Fn,
			  std::tuple_element_t<Ns, arg_tuple>>...>;
		return tuple_type{fn(std::get<Ns>(std::forward<Tpl>(tpl)))...};
	}
	else
		return swallow((fn(std::get<Ns>(std::forward<Tpl>(tpl))), 0)...);
}

struct map_functor {
	template <typename Fn, typename Tpl>
	constexpr auto
	operator()(Fn&& fn, Tpl&& tpl) const
	{
		using sequence = sequence_matching_tuple<Tpl>;
		return map(sequence(), std::forward<Fn>(fn), std::forward<Tpl>(tpl));
	}

	template <typename Fn, typename int_type, int_type ... ns>
	constexpr auto
	operator()(Fn&& fn, util::sequence<int_type, ns...>) const
	{
		constexpr auto id = [] (auto&& x) constexpr { return std::move(x); };
		using sequence = util::make_sequence<std::size_t, sizeof...(ns)>;
		return map(sequence(), id,
				std::forward_as_tuple(fn(std::integral_constant<int_type, ns>{})...));
	}
};

template <typename Fn, typename ... Args>
class partial {
private:
	using tuple_type = std::tuple<Args...>;
	static constexpr auto tuple_size = sizeof...(Args);
	static constexpr auto sequence = std::make_index_sequence<tuple_size>();
	Fn fn;
	tuple_type args;

public:
	template <typename ... Ts>
	constexpr auto
	operator()(Ts&& ... ts) const
	{
		std::tuple<Ts...> addl{std::forward<Ts>(ts)...};
		return apply(fn, std::tuple_cat(args, std::move(addl)));
	}

	constexpr partial(Fn&& fn, Args&& ... args) :
		fn(fn), args{std::forward<Args>(args)...} {}
};

struct bind_functor {
	template <typename Fn, typename ... Args>
	constexpr auto
	operator()(Fn&& fn, Args&& ... args) const
	{
		return std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
	}
};

struct partial_functor {
	template <typename Fn, typename ... Args>
	constexpr partial<Fn, Args...>
	operator()(Fn&& fn, Args&& ... args) const
	{
		return partial<Fn, Args...>(std::forward<Fn>(fn), std::forward<Args>(args)...);
	}
};

template <typename Fn, typename T0>
constexpr decltype(auto)
foldl(Fn&& fn, T0&& t0)
{
	return std::forward<T0>(t0);
}

template <typename Fn, typename T0, typename T1>
constexpr auto
foldl(Fn&& fn, T0&& t0, T1&& t1)
{
	return fn(std::forward<T0>(t0), std::forward<T1>(t1));
}

template <typename Fn, typename T0, typename T1, typename ... Ts>
constexpr auto
foldl(Fn&& fn, T0&& t0, T1&& t1, Ts&& ... ts)
{
	auto&& tn = fn(std::forward<T0>(t0), std::forward<T1>(t1));
	return foldl(std::forward<Fn>(fn), std::move(tn), std::forward<Ts>(ts)...);
}

struct foldl_functor {
	template <typename Fn, typename ... Ts>
	constexpr auto
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
constexpr auto
foldr(Fn&& fn, T0&& t0, T1&& t1)
{
	return fn(std::forward<T0>(t0), std::forward<T1>(t1));
}

template <typename Fn, typename T0, typename ... Ts>
constexpr auto
foldr(Fn&& fn, T0&& t0, Ts&& ... ts)
{
	return fn(std::forward<T0>(t0), foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...));
}

struct foldr_functor {
	template <typename Fn, typename ... Ts>
	constexpr auto
	operator()(Fn&& fn, Ts&& ... ts) const
	{
		return foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...);
	}
};

template <std::size_t ... Ns, typename Tpl>
constexpr auto
reverse(const std::index_sequence<Ns...>&, Tpl&& tpl)
{
	using tuple_type = std::decay_t<Tpl>;
	constexpr auto size = std::tuple_size_v<tuple_type>;
	using return_type = map_tuple<std::tuple_element_t<size-1-Ns, tuple_type>...>;
	return return_type{std::get<size-1-Ns>(std::forward<Tpl>(tpl))...};
}

struct reverse_functor {
	template <typename Tpl>
	constexpr auto
	operator()(Tpl&& tpl) const
	{
		return reverse(sequence_matching_tuple<Tpl>(), std::forward<Tpl>(tpl));
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

} // namespace functional
} // namespace util
