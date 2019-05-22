#pragma once
#include <type_traits>
#include <utility>

#include "util/launch.h"
#include "util/functional.h"

#include "types.h"
#include "operators.h"
#include "domain.h"
#include "identity.h"

namespace fd {
namespace boundary {
namespace impl {

using correction::order;

template <typename corrector_type, typename tag_type>
matrix
boundary(const corrector_type& corrector, const tag_type& tag)
{
	static constexpr auto solid_boundary = corrector_type::solid_boundary;
	const auto rows = corrector.points();
	const auto value = corrector.boundary_weight(tag);
	const auto row = tag_type::value ? rows-1 : 0;

	return single_entry(rows, solid_boundary, row, 0, value);
}

template <typename grid_type>
class boundary_builder {
private:
	static constexpr auto dimensions = std::tuple_size_v<grid_type>;
	using sequence = std::make_index_sequence<dimensions>;
	template <std::size_t n> using iteration = std::integral_constant<std::size_t, n>;
	template <std::size_t N> using order = boundary::correction::order<N>;

	template <std::size_t ... ns, typename views_type, typename view_type,
			 typename tag_type, std::size_t n>
	static auto
	splat(const std::index_sequence<ns...>&, const views_type& views,
			const view_type& view, const tag_type& tag, const order<n>& correction)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using corrector_type = boundary::corrector<colloc, dir_type>;
			corrector_type corrector(dir);

			return dir == view ?
				boundary(corrector, tag) :
				identity(corrector, correction);
		};

		return std::make_tuple(k(iteration<ns>{})...);
	}
public:
	template <typename Views, typename View, typename Tag, std::size_t N>
	static matrix
	build(const Views& views, const View& view, const Tag& tag, const order<N>& correction)
	{
		using namespace util::functional;
		auto unikron = [] (const matrix& l, const matrix& r) { return kron(l, r); };
		auto multikron = partial(foldl, unikron);
		auto&& components = splat(sequence(), views, view, tag, correction);
		auto&& reversed = reverse(std::move(components));
		return apply(multikron, std::move(reversed));
	}
};

} // namespace impl

template <typename Domain, typename View, typename Dir, std::size_t N = 0>
matrix
lower(const Domain& domain, const View& view, const Dir& dir,
	  const correction::order<N>& corr = correction::zeroth_order)
{
	using operators::caller;
	using impl::lower;
	using impl::boundary_builder;
	using tag_type = typename Domain::tag_type;
	using caller_type = caller<boundary_builder, tag_type>;
	auto&& views = fd::dimensions(domain);
	return caller_type::call(view, views, dir, lower, corr);
}

template <typename Domain, typename View, typename Dir, std::size_t N = 0>
matrix
upper(const Domain& domain, const View& view, const Dir& dir,
	  const correction::order<N>& corr = correction::zeroth_order)
{
	using operators::caller;
	using impl::upper;
	using impl::boundary_builder;
	using tag_type = typename Domain::tag_type;
	using caller_type = caller<boundary_builder, tag_type>;
	auto&& views = fd::dimensions(domain);
	return caller_type::call(view, views, dir, upper, corr);
}

} // namespace boundary
} // namespace fd
