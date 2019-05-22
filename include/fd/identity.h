#pragma once
#include <type_traits>
#include <utility>
#include <tuple>
#include "util/launch.h"
#include "util/functional.h"

#include "types.h"
#include "grid.h"
#include "correction.h"
#include "domain.h"
#include "operators.h"

namespace fd {
namespace impl {

using fd::boundary::correction::order;

template <typename corrector_type, std::size_t n>
matrix
identity(const corrector_type& corrector, const order<n>& correction)
{
	const auto rows = corrector.points();
	matrix result{rows, rows, rows};

	index_type* starts = result.starts();
	index_type* indices = result.indices();
	value_type* values = result.values();
	auto [lw, uw] = corrector.identity(correction);

	auto k = [=, lw = lw, uw = uw] __device__ (int tid)
	{
		starts[tid] = tid;
		indices[tid] = tid + indexing_base;
		values[tid] = 1.0 + (tid > 0 ? (tid < rows-1 ? 0 : uw) : lw);
		if (!tid) starts[rows] = rows;
	};
	util::transform<128, 7>(k, rows);
	return result;
}

template <typename Grid>
class identity_builder {
private:
	using grid_type = Grid;
	using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
	template <std::size_t n> using iteration = std::integral_constant<std::size_t, n>;

	template <std::size_t ... ns, typename Views, std::size_t n>
	static auto
	splat(const std::index_sequence<ns...>&, const Views& views,
			const order<n>& correction)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using corrector_type = boundary::corrector<colloc, dir_type>;
			corrector_type corrector(dir);
			return identity(corrector, correction);
		};

		return std::make_tuple(k(iteration<ns>{})...);
	}
public:
	template <typename Views, std::size_t N>
	static matrix
	build(const Views& views, const order<N>& correction)
	{
		using namespace util::functional;

		auto unikron = [] (const matrix& l, const matrix& r) { return kron(l, r); };
		auto multikron = partial(foldl, unikron);
		auto&& identities = splat(sequence(), views, correction);
		auto&& reversed = reverse(std::move(identities));
		return apply(multikron, std::move(reversed));
	}
};

} // namespace impl

template <typename domain_type, typename view_type, std::size_t n = 0,
		 typename = std::enable_if_t<is_domain_v<domain_type>>>
auto
identity(const domain_type& domain, const view_type& view,
		const boundary::correction::order<n>& correction = boundary::correction::zeroth_order)
{
	static_assert(is_domain_v<domain_type>);
	using operators::caller;
	using impl::identity_builder;
	using tag_type = typename domain_type::tag_type;
	using caller_type = caller<identity_builder, tag_type>;
	auto&& views = fd::dimensions(domain);
	return caller_type::call(view, views, correction);
}

template <typename domain_type, std::size_t n = 0,
		 typename = std::enable_if_t<is_domain_v<domain_type>>>
auto
identity(const domain_type& domain,
		const boundary::correction::order<n>& correction = boundary::correction::zeroth_order)
{
	static_assert(is_domain_v<domain_type>);
	static_assert(grid::is_uniform_v<domain_type::tag_type>,
			"the 2-argument variant of fd::identity requires a uniform grid (cell- or vertex-centered)");
	return identity(domain, std::get<0>(dimensions(domain)), correction);
}
} // namespace fd
