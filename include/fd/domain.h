#pragma once
#include <cstddef>
#include <array>
#include <tuple>
#include <utility>
#include <type_traits>
#include "algo/gcd.h"
#include "util/functional.h"
#include "types.h"
#include "grid.h"
#include "dimension.h"

namespace fd {

template <typename grid_tag, typename ... dimension_types>
class domain {
public:
	static_assert(is_grid_v<grid_tag>);
	static_assert((is_dimension_v<dimension_types> && ...));
	static constexpr auto ndim = sizeof...(dimension_types);
	using tag_type = grid_tag;
	using container_type = std::tuple<decltype(view{std::declval<dimension_types>(), 0, 0.0})...>;
protected:
	container_type _dimensions;
	double _resolution;
	index_type _refinement;

	constexpr domain(index_type refinement, double base,
			const dimension_types& ... dimensions) :
		_dimensions{view{dimensions, refinement, base}...},
		_resolution{refinement / base}, _refinement{refinement} {}
private:
	static constexpr auto gcd = util::functional::partial(
			util::functional::foldl, algo::gcd);
public:
	constexpr double resolution() const { return _resolution; }
	constexpr index_type refinement() const { return _refinement; }
	constexpr const container_type& dimensions() const { return _dimensions; }

	constexpr auto
	clamp(const std::array<double, ndim>& x) const
	{
		using namespace util::functional;
		auto k = [] (const auto& dim, double x) { return dim.clamp(x); };
		return map(partial(apply, k), zip(dimensions(), x));
	}

	constexpr domain(const tag_type& tag, const dimension_types& ... dimension) :
		domain(tag.refinement(), gcd((double) dimension.length()...), dimension...) {}

	template <typename old_tag>
	constexpr domain(const domain<old_tag, dimension_types...>& other) :
		_dimensions(other._dimensions),
		_refinement(other._refinement) {}
};

template <typename ... dimension_types>
domain(index_type, const dimension_types& ...) -> domain<fd::grid::mac, dimension_types...>;

template <typename> struct is_domain : std::false_type {};
template <typename grid_tag, typename ... dimension_types>
struct is_domain<domain<grid_tag, dimension_types...>> :
	std::integral_constant<bool, is_grid_v<grid_tag> && (is_dimension_v<dimension_types> && ...)> {};

template <typename domain_type>
inline constexpr auto is_domain_v = is_domain<domain_type>::value;

template <typename grid_tag, typename ... dimension_types>
constexpr auto
dimensions(const domain<grid_tag, dimension_types...>& dm)
{
	return dm.dimensions();
}

}
