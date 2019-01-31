#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>
#include "util/functional.h"
#include "grid.h"
#include "domain.h"
#include "operators.h"

namespace fd {
	namespace domain_size_impl {
		template <typename grid_type>
		class size_builder {
		private:
			using sequence = std::make_index_sequence<std::tuple_size_v<grid_type>>;
			template <std::size_t n> using collocation = std::tuple_element_t<n, grid_type>;

			template <typename collocation_type, typename view_type>
			static constexpr auto
			one(const view_type& view)
			{
				constexpr auto on_boundary = collocation_type::on_boundary;
				constexpr auto solid_boundary = view_type::solid_boundary;
				constexpr auto correction = solid_boundary * on_boundary;
				return view.gridpts() - correction;
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
				return std::make_tuple(one<collocation<ns>>(std::get<ns>(views))...);
			}

			template <std::size_t ... ns, typename views_type, typename view_type>
			static constexpr auto
			splat(const std::index_sequence<ns...>&, const views_type& views, const view_type& view)
			{
				return std::make_tuple(one<collocation<ns>>(std::get<ns>(views), view)...);
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
	}

	template <typename view_type, typename domain_type, typename ... view_types>
	constexpr auto
	sizes(const domain_type& domain, const view_type& view, const view_types& ... views)
	{
		static_assert(sizeof...(view_types) <= 1,
				"fd::sizes only accepts 2 or 3 parameters");
		using operators::caller;
		using tag_type = typename domain_type::tag_type;
		using domain_size_impl::size_builder;
		using caller_type = caller<size_builder, tag_type, 0, domain_type::ndim>;
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
}
