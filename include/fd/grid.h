#pragma once
#include <tuple>
#include <utility>
#include "lwps/types.h"

namespace fd {
	namespace grid_impl {
		struct center {
			static constexpr double shift = 0.5;
			static constexpr bool on_boundary = false;
		};

		struct edge {
			static constexpr double shift = 0.0;
			static constexpr bool on_boundary = true;
		};

		template <typename ... Cs>
		struct grid {
			static constexpr std::size_t ndim = sizeof...(Cs);
			template <std::size_t N> using collocation = std::tuple_element_t<N, std::tuple<Cs...>>;
		};

		template <typename, std::size_t, typename> struct grid_rule_evaluator;
		template <typename Tag, std::size_t I, std::size_t ... Ns>
		struct grid_rule_evaluator<Tag, I, std::index_sequence<Ns...>> {
			using type = grid<typename Tag::template rule<I, Ns>...>;
		};

		template <typename Tag, std::size_t I, std::size_t N>
		struct grid_maker {
			static constexpr std::size_t ndim = N;
			using type = typename grid_rule_evaluator<
				Tag, I, std::make_index_sequence<N>>::type;
		};

		template <typename> struct __cshift;
		template <> struct __cshift<center> { using type = edge; };
		template <> struct __cshift<edge> { using type = center; };
		template <typename T> using shifted = typename __cshift<T>::type;

		template <typename Tag, std::size_t, std::size_t J, std::size_t I>
		struct directionally {
			using type = typename Tag::template rule<I, J>;
		};

		template <typename Tag, std::size_t J, std::size_t I>
		struct directionally<Tag, J, I, I> {
			using type = shifted<typename Tag::template rule<I, J>>;
		};

		template <typename Tag, std::size_t I, std::size_t J>
		struct diagonally {
			using type = typename Tag::template rule<I, J>;
		};

		template <typename Tag, std::size_t I>
		struct diagonally<Tag, I, I> {
			using type = shifted<typename Tag::template rule<I, I>>;
		};

		class base_grid {
			private:
				lwps::index_type _resolution;
			public:
				constexpr const lwps::index_type& resolution() const { return _resolution; }
			protected:
				constexpr base_grid(lwps::index_type resolution) : _resolution(resolution) {}
		};

		template <typename Collocation>
		struct uniform : public base_grid {
			template <std::size_t, std::size_t> using rule = Collocation;

			static constexpr auto is_uniform = true;

			constexpr uniform(lwps::index_type resolution) : base_grid(resolution) {}
		};

		template <template <typename, std::size_t, std::size_t, std::size_t ...>
		          class Shifter, typename Tag, std::size_t ... Ns>
		struct shifted_tag : public base_grid {
			template <std::size_t I, std::size_t J>
			using rule = typename Shifter<Tag, I, J, Ns...>::type;

			static constexpr auto is_uniform = false;

			constexpr shifted_tag(lwps::index_type resolution) : base_grid(resolution) {}
		};

		template <template <typename, std::size_t, std::size_t, std::size_t ...> class S,
		          typename Tag, std::size_t ... Ns>
		struct tag_shifter {
			using type = shifted_tag<S, Tag, Ns...>;
		};

		template <template <typename, std::size_t, std::size_t, std::size_t ...> class S,
		          typename Tag, std::size_t ... Ns>
		struct tag_shifter<S, shifted_tag<S, Tag, Ns...>, Ns...> {
			using type = Tag;
		};

		template <template <typename, std::size_t, std::size_t, std::size_t ...> class S1,
		          template <typename, std::size_t, std::size_t, std::size_t ...> class S2,
		          typename Tag, std::size_t ... Ns1, std::size_t ... Ns2>
		struct tag_shifter<S1, shifted_tag<S2, Tag, Ns2...>, Ns1...> {
			using type = shifted_tag<S2, typename tag_shifter<S1, Tag, Ns1...>::type, Ns2...>;
		};
	}

	namespace grid {
		using grid_impl::center;
		using grid_impl::edge;

		namespace shift {
			template <typename T>
				using diagonally = typename grid_impl::tag_shifter<grid_impl::diagonally, T>::type;
			template <typename T, std::size_t N>
				using directionally = typename grid_impl::tag_shifter<grid_impl::directionally, T, N>::type;
		}

		using cell = grid_impl::uniform<center>;
		using mac = grid_impl::shifted_tag<grid_impl::diagonally, cell>;

		template <typename> struct is_uniform : std::false_type {};
		template <typename collocation_type> struct is_uniform<grid_impl::uniform<collocation_type>> : std::true_type {};
		template <typename grid_type> inline constexpr bool is_uniform_v = is_uniform<grid_type>::value;
	}
}

namespace std {
	template <size_t I, typename ... Cs>
	struct tuple_element<I, fd::grid_impl::grid<Cs...>> {
		using type = typename tuple_element<I, tuple<Cs...>>::type;
	};

	template <typename ... Cs>
	struct tuple_size<fd::grid_impl::grid<Cs...>> {
		static constexpr auto value = sizeof...(Cs);
	};
}
