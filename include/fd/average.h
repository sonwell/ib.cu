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
namespace impl {

template <typename corrector_type>
matrix
average(const corrector_type& corrector)
{
	static constexpr auto solid_boundary = corrector_type::solid_boundary;
	static constexpr auto on_boundary = corrector_type::on_boundary;
	static constexpr auto backward = solid_boundary ^ !on_boundary;

	const auto cells = corrector.cells();
	const auto cols = cells - solid_boundary * on_boundary;
	const auto rows = cells - solid_boundary * !on_boundary;

	auto nonzero = rows + cols - 1;
	matrix result{rows, cols, nonzero};

	// Periodic boundary
	// <-x-|-x-|-x-|-x-> --> x-o-x-o-x-o-x-o-> : backward, nxn, 2n-1 nonzero
	// x-o-x-o-x-o-x-o-> --> <-x-|-x-|-x-|-x-> : forward, nxn, 2n-1 nonzero
	//
	// Solid boundary
	// >-x-|-x-|-x-|-x-< --> >-o-x-o-x-o-x-o-< : forward, (n-1)xn, 2n-2 nonzero
	// >-o-x-o-x-o-x-o-< --> >-x-o-x-o-x-o-x-< : backward, nx(n-1), 2n-2 nonzero

	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();

	auto k = [=] __device__ (int tid)
	{
		if (tid < rows) starts[tid] = tid ? 2 * tid - backward : 0;
		auto loc = (tid + backward) % 2;
		auto row = (tid + backward) / 2;
		auto col = loc + row - backward;
		indices[tid] = col + indexing_base;
		values[tid] = 0.5;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	if constexpr (backward)
		result += corrector.interior(rows, cols, 0.5, boundary::lower);
	if constexpr (on_boundary)
		result += corrector.interior(rows, cols, 0.5, boundary::upper);
	return result;
}

template <typename Grid>
class average_builder {
private:
	using grid_type = Grid;
	using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
	template <std::size_t n> using iteration = std::integral_constant<std::size_t, n>;
	static constexpr auto& order = boundary::correction::zeroth_order;

	template <std::size_t ... ns, typename Views, typename view_type>
	static auto
	splat(const std::index_sequence<ns...>&, const Views& views, const view_type& view)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using corrector_type = boundary::corrector<colloc, dir_type>;
			corrector_type corrector(dir);

			return view == dir ?
				average(corrector) :
				identity(corrector, order);
		};

		return std::make_tuple(k(iteration<ns>{})...);
	}
public:
	template <typename Views, typename View>
	static matrix
	build(const Views& views, const View& view)
	{
		using namespace util::functional;

		auto unikron = [] (const matrix& l, const matrix& r) { return kron(l, r); };
		auto multikron = partial(foldl, unikron);
		auto&& components = splat(sequence(), views, view);
		auto&& reversed = reverse(std::move(components));
		return apply(multikron, std::move(reversed));
	}
};

} // namespace impl

template <typename domain_type, typename view_type, typename dir_type,
		 typename = std::enable_if_t<is_domain_v<domain_type>>>
auto
average(const domain_type& domain, const view_type& view, const dir_type& dir)
{
	using operators::caller;
	using impl::average_builder;
	using tag_type = typename domain_type::tag_type;
	using caller_type = caller<average_builder, tag_type>;
	auto&& views = fd::dimensions(domain);
	return caller_type::call(view, views, dir);
}

} // namespace fd
