#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>
#include "util/functional.h"
#include "grid.h"
#include "domain.h"
#include "discretization.h"
#include "operators.h"

namespace fd {
namespace impl {

template <typename grid_type>
class size_builder {
private:
	using sequence = std::make_index_sequence<std::tuple_size_v<grid_type>>;
	template <std::size_t n> using collocation = std::tuple_element_t<n, grid_type>;
	template <std::size_t n> using iteration = std::integral_constant<std::size_t, n>;

	template <typename info_type>
	static constexpr auto
	one(const info_type& info)
	{
		return info.points();
	}

	template <typename collocation_type, typename view_type, typename cmp_type>
	static constexpr auto
	one(const view_type& view, const cmp_type& cmp)
	{
		constexpr auto solid_boundary = view_type::solid_boundary;
		return view == cmp ? solid_boundary : one<collocation_type>(view);
	}

	template <std::size_t ... ns, typename views_type>
	static constexpr auto
	splat(const std::index_sequence<ns...>&, const views_type& views)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using info_type = discretization<colloc, dir_type>;
			info_type corrector(dir);
			return corrector.points();

		};
		return std::make_tuple(k(iteration<ns>{})...);
	}

	template <std::size_t ... ns, typename views_type, typename view_type>
	static constexpr auto
	splat(const std::index_sequence<ns...>&, const views_type& views, const view_type& view)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using info_type = discretization<colloc, dir_type>;
			info_type corrector(dir);

			return view == dir ?
				corrector.solid_boundary :
				corrector.points();

		};
		return std::make_tuple(k(iteration<ns>{})...);
	}
public:
	template <typename views_type>
	static constexpr auto
	build(const views_type& views)
	{
		return splat(sequence(), views);
	}

	template <typename views_type, typename view_type>
	static constexpr auto
	build(const views_type& views, const view_type& view)
	{
		return splat(sequence(), views, view);
	}
};

} // namespace impl

template <typename view_type, typename domain_type, typename ... view_types>
constexpr auto
sizes(const domain_type& domain, const view_type& view, const view_types& ... views)
{
	static_assert(sizeof...(view_types) <= 1,
			"fd::sizes only accepts 2 or 3 parameters");
	using operators::caller;
	using tag_type = typename domain_type::tag_type;
	using impl::size_builder;
	using caller_type = caller<size_builder, tag_type>;
	return caller_type::call(view, fd::dimensions(domain), views...);
}

template <typename domain_type, typename ... view_types>
constexpr auto
size(const domain_type& domain, const view_types& ... views)
{
	static_assert(sizeof...(view_types) >= 1 && sizeof...(view_types) <= 2,
			"fd::size only accepts 2 or 3 parameters");
	using namespace util::functional;
	auto&& sz = sizes(domain, views...);
	auto prod = partial(foldr, std::multiplies<int>(), 1u);
	return apply(prod, sz);
}

} // namespace fd
