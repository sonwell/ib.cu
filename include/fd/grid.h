#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "discretization.h"
#include "exceptions.h"

namespace fd {

// grid is the discretized equivalent to domain
template <typename> struct grid;

template <typename ... dimension_types>
struct grid<fd::domain<dimension_types...>> {
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
	using cell_type = std::array<fd::alignment, dimensions>;

	static constexpr auto
	discretizations(const domain_type& domain, double resolution, const cell_type& cell)
	{
		using namespace util::functional;
		const auto& comps = fd::components(domain);
		auto k = [&] (const auto& dimension, const fd::alignment& alignment)
		{
			return discretization{dimension, alignment, resolution};
		};
		return map(k, comps, cell);
	}

	using components_type = decltype(discretizations(std::declval<domain_type>(),
				0, std::declval<cell_type>()));
	typedef struct {
		const domain_type& domain;
		double resolution;
		components_type components;
	} info_type;

	static constexpr auto
	construct(const domain_type& domain, int ref, const cell_type& cell)
	{
		using namespace util::functional;
		auto base = domain.unit();
		auto res = ref / base;
		return info_type{domain, res, discretizations(domain, res, cell)};
	}

	template <typename tag_type, typename dimension_type>
	static constexpr auto
	cell(const tag_type&, const domain_type& domain, const dimension_type& dimension)
	{
		using init = std::integral_constant<std::size_t, 0>;
		const auto& comps = fd::components(domain);
		auto w = [&] (auto m) constexpr
		{
			constexpr auto i = decltype(m)::value;
			using cell_type = fd::cell_t<tag_type, i, dimensions>;
			return cell_type{}.alignments();
		};
		auto r = [&] (auto m, auto f) constexpr
		{
			constexpr auto i = decltype(m)::value;
			using next = std::integral_constant<decltype(i), i+1>;
			if constexpr (i < dimensions)
				return (dimension == std::get<i>(comps)) ? w(m) : f(next{}, f);
			else
				return (throw no_such_dimension(__PRETTY_FUNCTION__), w(init{}));
		};
		return r(init{}, r);
	}

	constexpr auto refinement() const { return _refinement; }
	constexpr auto resolution() const { return _resolution; }
	constexpr auto cell() const { return _cell; }
	constexpr auto components() const { return _components; }

	constexpr grid(const info_type& info, int ref, const cell_type& cell) :
		_domain(info.domain), _refinement(ref), _resolution(info.resolution),
		_cell(cell), _components(info.components) {}

	constexpr grid(const domain_type& domain, int ref, const cell_type& cell) :
		grid(construct(domain, ref, cell), ref, cell) {}

	constexpr grid(const grid& g, int ref) : grid(g._domain, ref, g.cell()) {}

	template <typename tag_type, typename dimension_type>
	constexpr grid(const tag_type& tag, const domain_type& domain,
			const dimension_type& dimension) :
		grid(domain, tag.refinement(), cell(tag, domain, dimension)) {}

	template <typename tag_type, typename = std::enable_if_t<is_uniform_v<tag_type>>>
	constexpr grid(const tag_type& tag, const domain_type& domain) :
		grid(tag, domain, std::get<0>(fd::components(domain))) {}

	const domain_type _domain;
	int _refinement;
	double _resolution;
	cell_type _cell;
	components_type _components;
};

template <typename tag_type, typename domain_type, typename dimension_type>
grid(const tag_type&, const domain_type&, const dimension_type&)
	-> grid<domain_type>;

template <typename tag_type, typename domain_type>
grid(const tag_type&, const domain_type&)
	-> grid<domain_type>;

template <typename domain_type>
grid(const grid<domain_type>&, int) -> grid<domain_type>;

template <typename domain_type>
constexpr decltype(auto)
components(const grid<domain_type>& grid)
{
	return grid.components();
}

template <typename> struct is_grid : std::false_type {};
template <typename domain_type>
struct is_grid<grid<domain_type>> : std::true_type {};

template <typename grid_type>
inline constexpr auto is_grid_v = is_grid<grid_type>::value;

} // namespace fd
