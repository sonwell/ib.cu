#pragma once
#include <tuple>

namespace fd {
	namespace operators_impl {
		class no_such_dimension : public std::runtime_error {
			public:
				no_such_dimension() : std::runtime_error("No such dimension.") {}
		};

		template <template <typename> class builder_template,
				 typename tag_type, std::size_t current, std::size_t maximum>
		struct caller {
			using grid_type = typename grid_impl::grid_maker<tag_type, current, maximum>::type;
			using builder_type = builder_template<grid_type>;

			template <typename view_type, typename views_type, typename ... arg_types>
			static constexpr auto
			call(const view_type& view, const views_type& views, arg_types&& ... args)
			{
				using next = caller<builder_template, tag_type, current+1, maximum>;
				return view == std::get<current>(views) ?
					builder_type::build(views, std::forward<arg_types>(args)...) :
					next::call(view, views, std::forward<arg_types>(args)...);
			}
		};

		template <template <typename> class builder_template,
				 typename tag_type, std::size_t maximum>
		struct caller<builder_template, tag_type, maximum, maximum> {
			using grid_type = typename grid_impl::grid_maker<tag_type, maximum, maximum>::type;
			using builder_type = builder_template<grid_type>;

			template <typename view_type, typename views_type, typename ... arg_types>
			static constexpr auto
			call(const view_type&, const views_type& views, arg_types&& ... args)
			{
				using return_type = decltype(builder_type::build(views, std::forward<arg_types>(args)...));
				return (throw no_such_dimension(), return_type{});
			}
		};
	}

	namespace operators {
		using operators_impl::caller;
		using operators_impl::no_such_dimension;
	}
}
