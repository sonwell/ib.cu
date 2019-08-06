#pragma once
//#include "differentiation.h"

namespace bases {
namespace detail {

/*struct empty : differentiable {
	typedef struct {
		static constexpr auto size() { return 0; }
		constexpr auto operator[](int) const { return 0; };
	} container;

	template <std::size_t n, int ... ds>
	constexpr auto
	operator()(const double (&)[n],
			partials<ds...> p = partials<>()) const
	{
		return container{};
	}
}; */

} // namespace detail

/*template <typename rbf_type, typename poly_type = detail::empty>
struct pair : differentiable {
	rbf_type rbf;
	poly_type poly;

	template <std::size_t n, int ... ds>
	constexpr auto
	operator()(const double (&xs)[n],
			partials<ds...> p = partials<>()) const
	{
		return diff(poly, p)(xs);
	}

	template <std::size_t n, int ... ds>
	constexpr auto
	operator()(const double (&xs)[n], const double (&xd)[n],
			partials<ds...> p = partials<>()) const
	{
		return diff(rbf, p)(xs, xd);
	}

	pair(rbf_type rbf, poly_type poly = detail::empty{}) :
		rbf(rbf), poly(poly) {}
}; */

template <typename data_type>
struct pair {
	data_type data;
	data_type sample;
};

} // namespace bases
