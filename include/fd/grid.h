#pragma once
#include <tuple>
#include <utility>
#include "util/array.h"
#include "util/math.h"
#include "types.h"

namespace fd {
namespace grid {

struct center {
	static constexpr double shift = 0.5;
	static constexpr bool on_boundary = false;
};

struct edge {
	static constexpr double shift = 0.0;
	static constexpr bool on_boundary = true;
};

namespace impl {

template <typename ... Cs>
struct grid {
	static constexpr std::size_t ndim = sizeof...(Cs);
	template <std::size_t N> using collocation = std::tuple_element_t<N, std::tuple<Cs...>>;
	static constexpr util::array<double, ndim> shifts = {Cs::shift...};
	static constexpr util::array<bool, ndim> on_boundary = {Cs::on_boundary...};
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

template <typename T>
struct shifted {
	static constexpr double shift = util::math::modulo(T::shift + 0.5, 1.0);
	static constexpr bool on_boundary = shift == 0.0;
	using type = shifted;
};

template <typename T>
struct shifted<shifted<T>> {
	using type = T;
};

template <typename T> using shifted_t = typename shifted<T>::type;

template <typename Tag, std::size_t, std::size_t J, std::size_t I>
struct directionally {
	using type = typename Tag::template rule<I, J>;
};

template <typename Tag, std::size_t J, std::size_t I>
struct directionally<Tag, J, I, I> {
	using type = shifted_t<typename Tag::template rule<I, J>>;
};

template <typename Tag, std::size_t I, std::size_t J>
struct diagonally {
	using type = typename Tag::template rule<I, J>;
};

template <typename Tag, std::size_t I>
struct diagonally<Tag, I, I> {
	using type = shifted_t<typename Tag::template rule<I, I>>;
};

class base_grid {
	private:
		index_type _refinement;
	public:
		constexpr const index_type& refinement() const { return _refinement; }
	protected:
		constexpr base_grid(index_type refinement) : _refinement(refinement) {}
};

template <typename Collocation>
struct uniform : public base_grid {
	template <std::size_t, std::size_t> using rule = Collocation;

	static constexpr auto is_uniform = true;

	constexpr uniform(index_type refinement) : base_grid(refinement) {}
};

template <template <typename, std::size_t, std::size_t, std::size_t ...>
		  class Shifter, typename Tag, std::size_t ... Ns>
struct shifted_tag : public base_grid {
	template <std::size_t I, std::size_t J>
	using rule = typename Shifter<Tag, I, J, Ns...>::type;

	static constexpr auto is_uniform = false;

	constexpr shifted_tag(index_type refinement) : base_grid(refinement) {}
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

} // namespace impl

namespace shift {

template <typename T>
	using diagonally = typename impl::tag_shifter<impl::diagonally, T>::type;
template <typename T, std::size_t N>
	using directionally = typename impl::tag_shifter<impl::directionally, T, N>::type;

} // namespace shift

using cell = impl::uniform<center>;
using mac = impl::shifted_tag<impl::diagonally, cell>;

template <typename> struct is_uniform : std::false_type {};
template <typename collocation_type> struct is_uniform<impl::uniform<collocation_type>> : std::true_type {};
template <typename grid_type> inline constexpr bool is_uniform_v = is_uniform<grid_type>::value;

template <typename Tag, std::size_t I, std::size_t N>
using make = typename impl::grid_maker<Tag, I, N>::type;

} // namespace grid

template <typename grid_type>
	struct is_grid : std::is_base_of<grid::impl::base_grid, grid_type> {};
template <typename grid_type>
inline constexpr auto is_grid_v = is_grid<grid_type>::value;

} // namespace fd

namespace std {

template <size_t I, typename ... Cs>
class tuple_element<I, fd::grid::impl::grid<Cs...>> {
public:
	using type = typename tuple_element<I, tuple<Cs...>>::type;
};

template <typename ... Cs>
class tuple_size<fd::grid::impl::grid<Cs...>> {
public:
	static constexpr auto value = sizeof...(Cs);
};

} // namespace std
