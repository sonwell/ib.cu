#pragma once
#include <type_traits>
#include <utility>

#include "util/launch.h"
#include "util/functional.h"

#include "types.h"
#include "operators.h"
#include "domain.h"
#include "identity.h"
#include "boundary.h"

namespace fd {
namespace impl {

template <typename corrector_type>
matrix
laplacian(const corrector_type& corrector)
{
	const auto rows = corrector.points();
	const auto n = corrector.resolution();
	if (!rows) return matrix{0, 0};

	auto nonzero = 3 * rows - 2;
	matrix result{rows, rows, nonzero};
	auto* starts = result.starts();
	auto* indices = result.indices();
	auto* values = result.values();
	const auto scale = n * n;

	auto k = [=] __device__ (int tid)
	{
		if (tid < rows) starts[tid] = tid ? 3 * tid - 1 : 0;
		auto loc = (tid + 1) % 3;
		auto row = (tid + 1) / 3;
		auto col = loc + row - 1;
		indices[tid] = col + indexing_base;
		values[tid] = (1 - 3 * (loc & 1)) * scale;
		if (!tid) starts[rows] = nonzero;
	};
	util::transform<128, 7>(k, nonzero);

	result += corrector.interior(rows, rows, scale, boundary::lower)
			+ corrector.interior(rows, rows, scale, boundary::upper);
	return result;
}

template <typename Grid>
class laplacian_builder {
private:
	using grid_type = Grid;
	using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
	template <std::size_t n> using iteration = std::integral_constant<std::size_t, n>;
	static constexpr auto& order = fd::boundary::correction::second_order;

	template <std::size_t ... ns, typename Views>
	static auto
	splat(const std::index_sequence<ns...>&, const Views& views)
	{
		auto k = [&] (auto m)
		{
			static constexpr auto id = decltype(m)::value;
			using colloc = std::tuple_element_t<id, grid_type>;
			const auto& dir = std::get<id>(views);
			using dir_type = std::decay_t<decltype(dir)>;
			using corrector_type = boundary::corrector<colloc, dir_type>;
			corrector_type corrector(dir);

			return std::pair{laplacian(corrector), identity(corrector, order)};
		};

		return std::make_tuple(k(iteration<ns>{})...);
	}
public:
	template <typename Views>
	static matrix
	build(const Views& views)
	{
		using namespace util::functional;

		static auto operation = [] (const auto& left, const auto& right)
		{
			auto&& l_lap = left.first;
			auto&& l_id = left.second;
			auto&& r_lap = right.first;
			auto&& r_id = right.second;
			return std::pair{kron(l_lap, r_id) + kron(l_id, r_lap), kron(l_id, r_id)};
		};

		auto&& pairs = splat(sequence(), views);
		auto&& multiop = partial(foldl, operation);
		auto&& reversed = reverse(std::move(pairs));
		auto&& results = apply(multiop, std::move(reversed));
		return results.first;
	}
};
} // namespace impl

template <typename domain_type, typename view_type,
		 typename = std::enable_if_t<is_domain_v<domain_type>>>
auto
laplacian(const domain_type& domain, const view_type& view)
{
	using operators::caller;
	using impl::laplacian_builder;
	using tag_type = typename domain_type::tag_type;
	using caller_type = caller<laplacian_builder, tag_type>;
	auto&& views = fd::dimensions(domain);
	return caller_type::call(view, views);
}

template <typename domain_type,
		 typename = std::enable_if_t<is_domain_v<domain_type>>>
auto
laplacian(const domain_type& domain)
{
	static_assert(grid::is_uniform_v<domain_type::tag_type>,
			"the 1-argument variant of fd::laplacian requires a uniform grid (cell- or vertex-centered)");
	return laplacian(domain, std::get<0>(dimensions(domain)));
}

} // namespace fd
