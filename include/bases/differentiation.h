#pragma once
#include "util/sequences.h"

namespace bases {
namespace detail {

template <typename, int ...> class counter;

template <int ... ns, int ... ms>
class counter<util::sequence<int, ns...>, ms...> {
private:
	static constexpr int sumeq(int) { return 0; }

	template <typename ... arg_types>
	static constexpr int
	sumeq(int n, int m, arg_types ... args)
	{
		return (n == m) + sumeq(n, args...);
	}
public:
	using type = util::sequence<int, sumeq(ns, ms...) ...>;
};

} // namespace detail

template <int ... ns>
struct partials {
	template <int n> using counts =
		typename detail::counter<util::make_sequence<int, n>, ns...>::type;

	using type = util::sequence<int, ns...>;
	constexpr partials(type) {}
	constexpr partials() {}
};

template <int ... ns, int ... ms>
constexpr auto
operator*(partials<ns...> l, partials<ms...> r)
{
	return partials{util::sort(util::sequence<int, ns..., ms...>())};
}


template <typename base, typename partials>
struct derivative : base {
	template <typename ... arg_types>
	constexpr auto
	operator()(arg_types&& ... args) const
	{
		return base::operator()(std::forward<arg_types>(args)..., partials{});
	}

	constexpr derivative(base fn, partials) : base(fn) {}
};

template <int n>
inline constexpr auto d = partials<n>{};

struct differentiable {};

template <typename base, typename wrt, int ... ds>
constexpr auto
diff(derivative<base, wrt> m, partials<ds...> p)
{
	return derivative{(base) m, wrt{} * p};
}

template <typename base, int ... ds,
		 typename = std::enable_if_t<std::is_base_of_v<differentiable, base>>>
constexpr auto
diff(base m, partials<ds...> p)
{
	return derivative{m, p};
}


} // namespace bases
