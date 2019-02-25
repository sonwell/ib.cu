#pragma once
#include <functional>
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include "debug.h"

namespace util {
	template <typename Tpl>
	using sequence_matching_tuple = std::make_index_sequence<std::tuple_size_v<std::decay_t<Tpl>>>;

	namespace functional_impl {
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
		struct is_invocable {
			using function_type = typename std::remove_reference<Fn>::type;
			template <typename U> static auto test(U* p) ->
				decltype((*p)(std::declval<Args>()...), void(), std::true_type());
			template <typename U> static auto test(...) -> std::false_type;

			static constexpr bool value = decltype(test<function_type>(nullptr))::value;
		};

		template <typename Fn, typename ... Args>
		struct __invoke_result {
			static_assert(is_invocable<Fn, Args...>::value,
			              "Function encountered unexpected argument(s)");
			using type = decltype(std::declval<Fn>()(std::declval<Args>()...));
		};

		template <typename Fn, typename ... Args>
		using invoke_result = typename __invoke_result<Fn, Args...>::type;

		template <typename Fn, typename Tpl, std::size_t ... Ns>
		constexpr auto
		__apply(const std::index_sequence<Ns...>&, Fn&& fn, Tpl&& args) ->
			invoke_result<Fn, decltype(std::get<Ns>(std::forward<Tpl>(args)))...>
		{
			return fn(std::get<Ns>(std::forward<Tpl>(args))...);
		}

		static constexpr struct __apply_functor {
			template <typename Fn, typename Tpl>
			constexpr auto
			operator()(Fn&& fn, Tpl&& args) const
			{
				using sequence = sequence_matching_tuple<Tpl>;
				return __apply(sequence(), std::forward<Fn>(fn), std::forward<Tpl>(args));
			}
		} apply;

		template <std::size_t N, typename ... Ts>
		constexpr auto
		__zip(Ts&& ... ts)
		{
			return std::forward_as_tuple(std::get<N>(std::forward<Ts>(ts))...);
		}

		static constexpr struct __zip_functor {
			private:
				template <std::size_t ... Ns, typename ... Tpls>
				static constexpr auto
				call(const std::index_sequence<Ns...>&, Tpls&& ... tpls)
				{
					return std::make_tuple(__zip<Ns>(std::forward<Tpls>(tpls)...)...);
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
		} zip;

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

		void __swallow(...) {}

		template <std::size_t ... Ns, typename Fn, typename Tpl>
		constexpr auto
		__map(const std::index_sequence<Ns...>&, Fn&& fn, Tpl&& tpl)
		{
			using arg_tuple = std::decay_t<Tpl>;
			constexpr bool returns_void = any_v<std::is_void_v<invoke_result<Fn, std::tuple_element_t<Ns, arg_tuple>>>...>;
			if constexpr (!returns_void) {
				using tuple_type = map_tuple<invoke_result<Fn, std::tuple_element_t<Ns, arg_tuple>>...>;
				return tuple_type{fn(std::get<Ns>(std::forward<Tpl>(tpl)))...};
			}
			else
				return __swallow((fn(std::get<Ns>(std::forward<Tpl>(tpl))), 0)...);
		}

		static constexpr struct __map_functor {
			template <typename Fn, typename Tpl>
			constexpr auto
			operator()(Fn&& fn, Tpl&& tpl) const
			{
				using sequence = sequence_matching_tuple<Tpl>;
				return __map(sequence(), std::forward<Fn>(fn), std::forward<Tpl>(tpl));
			}
		} map;

		template <typename Fn, typename ... Args>
		class __partial {
			private:
				using tuple_type = std::tuple<Args...>;
				static constexpr auto tuple_size = sizeof...(Args);
				static constexpr auto sequence = std::make_index_sequence<tuple_size>();
				Fn _fn;
				tuple_type _args;

			public:
				template <typename ... Ts>
				constexpr auto
				operator()(Ts&& ... ts) const
				{
					std::tuple<Ts...> addl{std::forward<Ts>(ts)...};
					return apply(_fn, std::tuple_cat(_args, std::move(addl)));
				}

				constexpr __partial(Fn fn, Args&& ... args) :
					_fn(fn), _args{std::forward<Args>(args)...} {}
		};

		static constexpr struct __bind_functor {
			template <typename Fn, typename ... Args>
			constexpr auto
			operator()(Fn&& fn, Args&& ... args) const
			{
				return std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
			}
		} bind;

		static constexpr struct __partial_functor {
			template <typename Fn, typename ... Args>
			constexpr __partial<Fn, Args...>
			operator()(Fn fn, Args&& ... args) const
			{
				return __partial<Fn, Args...>(fn, std::forward<Args>(args)...);
			}
		} partial;

		template <typename Fn, typename T0, typename T1, typename ... Ts>
		struct __foldl_result : __foldl_result<Fn, typename __invoke_result<Fn, T0, T1>::type, Ts...> {};

		template <typename Fn, typename T0, typename T1>
		struct __foldl_result<Fn, T0, T1> : __invoke_result<Fn, T0, T1> {};

		template <typename Fn, typename T0, typename ... Ts>
		using foldl_result = typename __foldl_result<Fn, T0, Ts...>::type;

		template <typename Fn, typename T0>
		constexpr decltype(auto)
		__foldl(Fn&& fn, T0&& t0)
		{
			return std::forward<T0>(t0);
		}

		template <typename Fn, typename T0, typename T1>
		constexpr foldl_result<Fn, T0, T1>
		__foldl(Fn&& fn, T0&& t0, T1&& t1)
		{
			return fn(std::forward<T0>(t0), std::forward<T1>(t1));
		}

		template <typename Fn, typename T0, typename T1, typename ... Ts>
		constexpr foldl_result<Fn, T0, T1, Ts...>
		__foldl(Fn&& fn, T0&& t0, T1&& t1, Ts&& ... ts)
		{
			return __foldl(std::forward<Fn>(fn), fn(std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...);
		}

		static constexpr struct __foldl_functor {
			template <typename Fn, typename ... Ts>
			constexpr auto
			operator()(Fn&& fn, Ts&& ... ts) const
			{
				return __foldl(std::forward<Fn>(fn), std::forward<Ts>(ts)...);
			}
		} foldl;

		template <typename Fn, typename T0, typename ... Ts>
		struct __foldr_result : __invoke_result<Fn, T0, typename __foldr_result<Fn, Ts...>::type> {};

		template <typename Fn, typename T0>
		struct __foldr_result<Fn, T0> { using type = T0; };

		template <typename Fn, typename T0, typename T1>
		struct __foldr_result<Fn, T0, T1> : __invoke_result<Fn, T0, T1> {};

		template <typename Fn, typename T0, typename ... Ts>
		using foldr_result = typename __foldr_result<Fn, T0, Ts...>::type;

		template <typename Fn, typename T0>
		constexpr decltype(auto)
		__foldr(Fn&& fn, T0&& t0)
		{
			return std::forward<T0>(t0);
		}

		template <typename Fn, typename T0, typename T1>
		constexpr foldr_result<Fn, T0, T1>
		__foldr(Fn&& fn, T0&& t0, T1&& t1)
		{
			return fn(std::forward<T0>(t0), std::forward<T1>(t1));
		}

		template <typename Fn, typename T0, typename ... Ts>
		constexpr foldr_result<Fn, T0, Ts...>
		__foldr(Fn&& fn, T0&& t0, Ts&& ... ts)
		{
			return fn(std::forward<T0>(t0), __foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...));
		}

		static constexpr struct __foldr_functor {
			template <typename Fn, typename ... Ts>
			constexpr auto
			operator()(Fn&& fn, Ts&& ... ts) const
			{
				return __foldr(std::forward<Fn>(fn), std::forward<Ts>(ts)...);
			}
		} foldr;

		template <std::size_t ... Ns, typename Tpl>
		constexpr auto
		__reverse(const std::index_sequence<Ns...>&, Tpl&& tpl)
		{
			using tuple_type = std::decay_t<Tpl>;
			constexpr auto size = std::tuple_size_v<tuple_type>;
			using return_type = map_tuple<std::tuple_element_t<size-1-Ns, tuple_type>...>;
			return return_type{std::get<size-1-Ns>(std::forward<Tpl>(tpl))...};
		}

		static constexpr struct __reverse_functor {
			template <typename Tpl>
			constexpr auto
			operator()(Tpl&& tpl) const
			{
				return __reverse(sequence_matching_tuple<Tpl>(), std::forward<Tpl>(tpl));
			}
		} reverse;
	}

	namespace functional {
		using functional_impl::zip;
		using functional_impl::apply;
		using functional_impl::map;
		using functional_impl::bind;
		using functional_impl::partial;
		using functional_impl::foldr;
		using functional_impl::foldl;
		using functional_impl::reverse;
	}
}
