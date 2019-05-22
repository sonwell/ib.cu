#pragma once
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include "util/counter.h"
#include "util/math.h"
#include "units.h"
#include "types.h"
#include "exceptions.h"
#include "boundary.h"

namespace fd {
namespace impl {

// a bit of misdirection for the constexpr counter
// see note in include/util/counter.h
struct counter_base { static constexpr util::counter<unsigned> value; };
template <unsigned> struct counter : counter_base { using counter_base::value; };

class base_dimension {
private:
	const unsigned _id;
	const units::distance _size;
public:
	constexpr auto id() const { return _id; }
	constexpr auto length() const { return _size; }
	constexpr bool operator==(const base_dimension& other) const
		{ return _id == other._id; }
	constexpr bool operator!=(const base_dimension& other) const
		{ return _id != other._id; }
protected:
	constexpr base_dimension(units::distance size, unsigned id) :
		_id(id), _size(size)
	{
		if ((double) size < 0)
			throw bad_grid_points("dimension size must be positive");
	}
};

} // namespace impl

template <typename lower_bdy, typename upper_bdy = lower_bdy>
class dimension : public impl::base_dimension {
static_assert(boundary::is_valid_combination_v<lower_bdy, upper_bdy>,
			"A dimension must either be periodic at both ends or neither end");
public:
	using lower_boundary_type = lower_bdy;
	using upper_boundary_type = upper_bdy;

	static constexpr bool solid_boundary = !std::is_same_v<
		lower_boundary_type, boundary::periodic>;

	constexpr const auto& lower() const { return _lower; }
	constexpr const auto& upper() const { return _upper; }
	constexpr double
	clamp(double x) const
	{
		auto length = impl::base_dimension::length();
		if constexpr (solid_boundary)
			return util::math::max(0, util::math::min(length, x));
		return util::math::modulo(x, length);
	}

	template <unsigned n = 0, unsigned id = next(impl::counter<n>::value)>
	constexpr dimension(units::distance size, const lower_boundary_type& lower,
			const upper_boundary_type& upper) :
		dimension(size, lower, upper, id) {}

	constexpr dimension(units::distance size, const lower_boundary_type& lower) :
		dimension(size, lower, lower)
	{
		static_assert(std::is_same_v<lower_boundary_type, upper_boundary_type>,
				"dimension constructor requires 2 boundaries");
	}

	template <typename old_lower, typename old_upper>
	constexpr dimension(const dimension<old_lower, old_upper>& other,
			const lower_boundary_type& lower,
			const upper_boundary_type& upper) :
		base_dimension(other), _lower(lower), _upper(upper) {}

	template <typename old_lower, typename old_upper>
	constexpr dimension(const dimension<old_lower, old_upper>& other,
			const lower_boundary_type& lower) :
		base_dimension(other), _lower(lower), _upper(lower)
	{
		static_assert(std::is_same_v<lower_boundary_type, upper_boundary_type>,
				"dimension constructor requires 2 boundaries");
	}
private:
	const lower_boundary_type _lower;
	const upper_boundary_type _upper;

	constexpr dimension(units::distance size, const lower_boundary_type& lower,
			const upper_boundary_type& upper, unsigned id) :
		base_dimension(size, id), _lower(lower), _upper(upper) {}
};

// Deduction help for single-parameter-type boundary
template <typename boundary_type>
dimension(units::distance, const boundary_type&) -> dimension<boundary_type, boundary_type>;

template <typename old_lower, typename old_upper, typename boundary_type>
dimension(const dimension<old_lower, old_upper>&, const boundary_type&) ->
	dimension<boundary_type, boundary_type>;

template <typename lower, typename upper>
class view : public dimension<lower, upper> {
private:
	using dimension_type = dimension<lower, upper>;
	const double _resolution;
	const int _cells;
public:
	constexpr auto cells() const { return _cells; }
	constexpr auto resolution() const { return _resolution; }

	constexpr view(const dimension_type& dimension, index_type refinement, double base) :
		dimension_type(dimension), _resolution(refinement / base),
		// add 0.5 to ensure we get the correct integer result;
		// length() / base should be (close to) an integer.
		_cells((double) dimension_type::length() * _resolution + 0.5) {}
};

template <typename lower, typename upper>
view(const dimension<lower, upper>&, index_type, double) -> view<lower, upper>;

template <typename dimension_type>
struct is_dimension : std::is_base_of<impl::base_dimension, dimension_type> {};
template <typename dimension_type>
inline constexpr auto is_dimension_v = is_dimension<dimension_type>::value;

} // namespace fd
