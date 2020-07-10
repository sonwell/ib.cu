#pragma once
#include <tuple>
#include <array>
#include <utility>
#include "util/math.h"
#include "types.h"

namespace fd {

struct alignment {
	constexpr auto on_boundary() const { return !shift; }

	double shift;

	constexpr alignment(double shift) :
		shift(util::math::modulo(shift, 1.0)) {}
};

// A cell is defined by is location in units of h relative to the origin. The
// alignments (aka staggering) of a grid point (the lower corner of the cell) is
// given by (x mod h) / h and is typically some combination of 0s and 0.5s. For
// a solid boundary, the staggering away from the boundary is greater than 0,
// but a staggering of 1 means that there is a "grid point" on the boundary, so
// we will write 0, but in this case 0 and 1 mean the same thing, in some sense.
template <typename ... alignment_types>
struct cell {
public:
	static constexpr auto dimensions = sizeof...(alignment_types);

	static constexpr auto
	alignments()
	{
		using array_type = std::array<alignment, dimensions>;
		return array_type{alignment{alignment_types::shift}...};
	}

	constexpr auto operator[](int n) const { return alignments()[n]; }
};

namespace __1 {

// Staggering is 0.5
struct center {
	static constexpr double shift = 0.5;
	static constexpr bool on_boundary = false;
};

// Staggering is 0
struct edge {
	static constexpr double shift = 0.0;
	static constexpr bool on_boundary = true;
};

// We can modify uniformly cell-centered or vertex-centered grids by shifting
// according to some rules. Diagonally means the i-th grid is shifted by 0.5 in
// the i-th dimension. The standard grid staggering is to diagonally shift
// cell-centered grids:
//
//     (0.5, 0.5) -> (0, 0.5)  x grid
//     (0.5, 0.5) -> (0.5, 0)  y grid
//
// Some operators act in a directional manner: averaging, and differentiation,
// for example. Their effect is to approximate the desired quantity at a shifted
// location. This is shifting directionally:
//
//     (0, 0.5) -x-shift-> (0.5, 0.5)  x grid
//     (0.5, 0) -x-shift-> (0, 0)      y grid
template <typename, std::size_t, typename> struct rule_evaluator;
template <typename tag, std::size_t i, std::size_t ... n>
struct rule_evaluator<tag, i, std::index_sequence<n...>> {
	using type = cell<typename tag::template rule<i, n>...>;
};

template <typename tag, std::size_t i, std::size_t n>
struct cell_maker {
	using sequence = std::make_index_sequence<n>;
	using type = typename rule_evaluator<tag, i, sequence>::type;
};

template <typename alignment_type>
struct shifted {
	static constexpr double shift =
		util::math::modulo(alignment_type::shift + 0.5, 1.0);
	static constexpr bool on_boundary = !shift;
	using type = shifted;
};

template <typename alignment_type>
struct shifted<shifted<alignment_type>> {
	using type = alignment_type;
};

template <typename alignment_type>
using shifted_t = typename shifted<alignment_type>::type;

template <typename tag, std::size_t, std::size_t j, std::size_t i>
struct directionally {
	using type = typename tag::template rule<i, j>;
};

template <typename tag, std::size_t j, std::size_t i>
struct directionally<tag, j, i, i> {
	using type = shifted_t<typename tag::template rule<i, j>>;
};

template <typename tag, std::size_t i, std::size_t j>
struct diagonally {
	using type = typename tag::template rule<i, j>;
};

template <typename tag, std::size_t i>
struct diagonally<tag, i, i> {
	using type = shifted_t<typename tag::template rule<i, i>>;
};

class cell_base {
public:
	constexpr int refinement() const { return ref; }
protected:
	constexpr cell_base(int ref) : ref(ref) {}
	constexpr cell_base(const cell_base& base) :
		cell_base(base.refinement()) {}
private:
	int ref;
};

template <typename alignment_type>
struct uniform : public cell_base {
	template <std::size_t, std::size_t> using rule = alignment_type;
	constexpr uniform(int ref) : cell_base(ref) {}
	constexpr uniform(const cell_base& base) : cell_base(base) {}
};

template <template <typename, std::size_t, std::size_t, std::size_t ...> class ruleset,
		  typename tag, std::size_t ... n>
struct shifted_tag : public cell_base {
	template <std::size_t i, std::size_t j>
	using rule = typename ruleset<tag, i, j, n...>::type;
	constexpr shifted_tag(int ref) : cell_base(ref) {}
	constexpr shifted_tag(const cell_base& base) : cell_base(base) {}
};

template <template <typename, std::size_t, std::size_t, std::size_t ...> class ruleset,
		  typename tag, std::size_t ... n>
struct tag_shifter {
	using type = shifted_tag<ruleset, tag, n...>;
};

template <template <typename, std::size_t, std::size_t, std::size_t ...> class ruleset,
		  typename tag, std::size_t ... n>
struct tag_shifter<ruleset, shifted_tag<ruleset, tag, n...>, n...> {
	using type = tag;
};

template <template <typename, std::size_t, std::size_t, std::size_t ...> class ruleset1,
		  template <typename, std::size_t, std::size_t, std::size_t ...> class ruleset2,
		  typename tag, std::size_t ... n, std::size_t ... m>
struct tag_shifter<ruleset1, shifted_tag<ruleset2, tag, m...>, n...> {
	using recurse_type = typename tag_shifter<ruleset1, tag, n...>::type;
	using type = shifted_tag<ruleset2, recurse_type, m...>;
};

} // namespace __1

namespace shift {

template <typename T>
	using diagonal = typename __1::tag_shifter<__1::diagonally, T>::type;
template <typename T, std::size_t n>
	using directional = typename __1::tag_shifter<__1::directionally, T, n>::type;

template <typename T>
constexpr auto
diagonally(const T& t)
{
	return diagonal<T>{t.refinement()};
}

template <std::size_t n, typename T>
constexpr auto
directionally(const T& t)
{
	return directional<T, n>{t.refinement()};
}

constexpr auto
alignment(const fd::alignment& alignment)
{
	return fd::alignment{util::math::modulo(alignment.shift + 0.5, 1.0)};
}

} // namespace shift

using centered = __1::uniform<__1::center>;
using mac = __1::shifted_tag<__1::diagonally, centered>;

template <typename> struct is_uniform : std::false_type {};
template <typename alignment_type> struct is_uniform<__1::uniform<alignment_type>> : std::true_type {};
template <typename cell_type> inline constexpr bool is_uniform_v = is_uniform<cell_type>::value;

template <typename tag, std::size_t i, std::size_t n>
using cell_t = typename __1::cell_maker<tag, i, n>::type;

template <typename cell_type>
struct is_cell : std::is_base_of<__1::cell_base, cell_type> {};

template <typename cell_type>
inline constexpr auto is_cell_v = is_cell<cell_type>::value;

} // namespace fd

namespace std {

template <size_t i, typename ... alignment_types>
class tuple_element<i, fd::cell<alignment_types...>> {
public:
	using type = typename tuple_element<i, tuple<alignment_types...>>::type;
};

template <typename ... alignment_types>
class tuple_size<fd::cell<alignment_types...>> {
public:
	static constexpr auto value = sizeof...(alignment_types);
};

} // namespace std
