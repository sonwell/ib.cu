#pragma once
#include <array>
#include "util/wrapper.h"
#include "util/functional.h"
#include "discretization.h"
#include "exceptions.h"

namespace fd {
namespace __1 {

template <typename dst_type, typename src_type>
constexpr decltype(auto)
assign(dst_type&& dst, src_type&& src)
{
	using namespace util::functional;
	auto k = [] (auto& d, auto&& s) { d = s; };
	map(k, dst, std::forward<src_type>(src));
	return std::move(dst);
}

template <std::size_t dimensions>
using point = util::wrapper<struct point_tag, std::array<double, dimensions>>;

template <std::size_t dimensions>
using delta = util::wrapper<struct delta_tag, std::array<double, dimensions>>;

template <std::size_t dimensions>
using units = util::wrapper<struct units_tag, std::array<double, dimensions>>;

template <std::size_t dimensions>
using indices = util::wrapper<struct indices_tag, std::array<int, dimensions>>;

template <std::size_t dimensions>
using shift = util::wrapper<struct shift_tag, std::array<int, dimensions>>;

template <std::size_t dimensions>
constexpr auto
operator+(indices<dimensions> ind, const shift<dimensions>& sh)
{
	using namespace util::functional;
	auto k = [] (int& i, const int& j) { i += j; };
	map(k, ind, sh);
	return ind;
}

template <std::size_t dimensions>
constexpr decltype(auto)
operator-(const units<dimensions>& left, const units<dimensions>& right)
{
	using namespace util::functional;
	auto k = [] (const double& l, const double& r) { return l - r; };
	return assign(delta<dimensions>{0}, map(k, left, right));
}

template <std::size_t dimensions>
constexpr decltype(auto)
operator+(units<dimensions> left, const delta<dimensions>& right)
{
	using namespace util::functional;
	auto k = [] (double& l, const double& r) { l += r; };
	map(k, left, right);
	return left;
}

template <typename cell_type>
struct cell_builder {
	template <typename views_type>
	static constexpr auto
	build(const views_type&)
	{
		return cell_type{}.alignments();
	}
};

struct container {
	int index;
	int lower;
	int upper;

	constexpr container&
	operator*=(const container& o)
	{
		auto k = [] (const container& c)
		{
			return c.lower <= c.index && c.index <  c.upper ?
				c.index : c.lower - 1;
		};
		auto weight = upper - lower;
		index = k(*this) + weight * k(o);
		lower = weight * o.lower;
		upper = weight * o.upper;
		return *this;
	}

	constexpr int value() const { return index; }
};

constexpr container
operator*(container left, const container& right)
{
	left *= right;
	return left;
}

template <typename> struct grid;

template <typename ... dimension_types>
struct grid<fd::domain<dimension_types...>> {
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
	using cell_type = std::array<fd::alignment, dimensions>;
	using point_type = point<dimensions>;
	using units_type = units<dimensions>;
	using indices_type = indices<dimensions>;
	using delta_type = delta<dimensions>;

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
				return (throw no_such_dimension(__PRETTY_FUNCTION__),
						std::declval<decltype(w(init{}))>());
		};
		return r(init{}, r);
	}

	constexpr auto refinement() const { return _refinement; }
	constexpr auto resolution() const { return _resolution; }
	constexpr auto cell() const { return _cell; }
	constexpr auto components() const { return _components; }

	constexpr auto
	units(const point_type& z) const
	{
		using namespace util::functional;
		auto k = [] (const auto& comp, double x) { return comp.units(x); };
		return assign(units_type{0.0}, map(k, _components, z));
	};

	constexpr auto
	difference(units_type z) const
	{
		using namespace util::functional;
		auto k = [] (double& u, const double& v, const auto& comp)
		{
			auto shift = comp.shift();
			auto diff = v - shift;
			u = util::math::floor(diff) - diff;
		};

		delta_type w = {0.0};
		map(k, w, z, components());
		return w;
	}

	template <typename point_type, typename indexer_type>
	constexpr auto
	index(const point_type& u, indexer_type&& indexer) const
	{
		using namespace util::functional;

		auto op = [] (const auto& l, const auto& r)
		{
			return l * r;
			/*auto [li, ll, lu] = l;
			auto [ri, rl, ru] = r;
			auto lw = lu - ll;
			auto lj = ll <= li && li < lu ? li : ll-1;
			auto rj = rl <= ri && ri < ru ? ri : rl-1;
			return container{lj + lw * rj, lw * rl, lw * ru};*/
		};

		auto info = map(indexer, u, components());
		return apply(partial(foldl, op), info).value();
	}

	constexpr auto
	index(const indices_type& u) const
	{
		auto k = [] (int i, const auto& comp)
		{
			auto j = comp.index(i);
			return container{j, 0, comp.points()};
		};
		return index(u, k);
	}

	constexpr auto
	index(const units_type& u) const
	{
		auto k = [] (double v, const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto shift = comp.shift();
			auto i = (int) util::math::floor(v - shift);
			return container{i, -solid, comp.points()};
		};
		return index(u, k);
	}

	constexpr auto
	indices(int index) const
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp)
		{
			auto solid = comp.solid_boundary;
			auto buffer = comp.points() + solid;
			auto u = (index + solid) % buffer - solid;
			index /= buffer;
			return u;
		};
		return assign(indices_type{0}, map(k, components()));
	}

	constexpr auto
	point(int index) const
	{
		auto j = indices(index);
		auto k = [] (int i, const auto& comp) { return comp.point(i); };
		return assign(point_type{0.}, map(k, j, components()));
	}

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

} // namespace __1

using __1::grid;

template <typename> struct is_grid : std::false_type {};
template <typename domain_type>
struct is_grid<grid<domain_type>> : std::true_type {};

template <typename grid_type>
inline constexpr auto is_grid_v = is_grid<grid_type>::value;

} // namespace ib
