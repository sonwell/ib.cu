#pragma once
#include <tuple>
#include "grid.h"
#include "exceptions.h"

namespace fd {
namespace operators {

template <template <typename> class tmpl, typename tag_type, std::size_t current = 0>
struct caller {
	template <typename view_type, typename views_type, typename ... arg_types>
	static constexpr auto
	call(const view_type& view, const views_type& views, arg_types&& ... args)
	{
		constexpr auto maximum = std::tuple_size_v<views_type>;
		using builder = tmpl<grid::make<tag_type, current, maximum>>;

		using built_type = decltype(builder::build(views, args...));
		using next = caller<tmpl, tag_type, current+1>;
		if constexpr (current < maximum)
			return view == std::get<current>(views) ?
				builder::build(views, std::forward<arg_types>(args)...) :
				next::call(view, views, std::forward<arg_types>(args)...);
		else
			return (throw no_such_dimension(__PRETTY_FUNCTION__), built_type{});
	}
};

} // namespace operators
} // namespace fd
