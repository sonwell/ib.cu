#pragma once
#include <cstddef>
#include <tuple>
#include <utility>
#include "grid.h"
#include "dimension.h"
#include "lwps/types.h"

namespace fd {
	template <typename Tag, typename ... Ds>
	class domain {
	public:
		static constexpr auto ndim = sizeof...(Ds);
		using tag_type = Tag;
		using container_type = std::tuple<
			decltype(dimension_impl::view{0, std::declval<Ds>()})...>;
	protected:
		lwps::index_type _resolution;
		container_type _dimensions;
	public:
		constexpr std::size_t resolution() const { return _resolution; }
		constexpr const container_type& dimensions() const { return _dimensions; }

		constexpr domain(const tag_type& tag, const Ds& ... ds) :
			_resolution{tag.resolution()},
			_dimensions{dimension_impl::view{tag.resolution(), ds}...} {}

		template <typename OldTag>
		constexpr domain(const domain<OldTag, Ds...>& other) :
			_resolution(other._resolution),
			_dimensions(other._dimensions) {}
	};

	template <typename Tag, typename ... Ds>
	constexpr auto
	dimensions(const domain<Tag, Ds...>& dm)
	{
		return dm.dimensions();
	}
}
