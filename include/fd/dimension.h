#pragma once
#include <cstddef>
#include <stdexcept>
#include "util/counter.h"
#include "util/math.h"
#include "units.h"
#include "types.h"
#include "exceptions.h"
#include "boundary.h"

namespace fd {
namespace __1 {

// a bit of misdirection for the constexpr counter
// see note in include/util/counter.h
struct counter_base { static constexpr util::counter<unsigned> value; };
template <unsigned> struct counter : counter_base { using counter_base::value; };

class dimension_base {
private:
	unsigned _id;
	units::length _length;
public:
	//static constexpr auto dimensions = 1;
	constexpr auto id() const { return _id; }
	constexpr auto length() const { return _length; }
	constexpr bool operator==(const dimension_base& other) const
		{ return _id == other._id; }
	constexpr bool operator!=(const dimension_base& other) const
		{ return _id != other._id; }
protected:
	constexpr dimension_base(units::length size, unsigned id) :
		_id(id), _length(size)
	{
		if ((double) size < 0)
			throw bad_grid_points("dimension size must be positive");
	}
};

template <typename lower_bdy, typename upper_bdy = lower_bdy>
class dimension : public dimension_base {
static_assert(boundary::is_valid_combination_v<lower_bdy, upper_bdy>,
			"A dimension must either be periodic at both ends or neither end");
public:
	using lower_boundary_type = lower_bdy;
	using upper_boundary_type = upper_bdy;
private:
	static constexpr bool single_boundary_enabled = std::is_same_v<
		lower_boundary_type, upper_boundary_type>;
public:
	static constexpr bool solid_boundary = lower_boundary_type::solid;

	constexpr const auto& lower() const { return _lower; }
	constexpr const auto& upper() const { return _upper; }

	constexpr double
	clamp(double x) const
	{
		if constexpr (solid_boundary) return x;
		return util::math::modulo(x, length());
	}

	template <unsigned n = 0, unsigned id = next(counter<n>::value)>
	constexpr dimension(units::length size, const lower_boundary_type& lower,
			const upper_boundary_type& upper) :
		dimension(size, lower, upper, id) {}

	template <unsigned n = 0,
		typename = std::enable_if_t<single_boundary_enabled>,
		unsigned id = next(counter<n>::value)>
	constexpr dimension(units::length size, const lower_boundary_type& lower) :
		dimension(size, lower, lower, id) {}

	template <typename old_lower, typename old_upper>
	constexpr dimension(const dimension<old_lower, old_upper>& other,
			const lower_boundary_type& lower,
			const upper_boundary_type& upper) :
		dimension_base(other), _lower(lower), _upper(upper) {}

	template <typename old_lower, typename old_upper,
			 typename = std::enable_if_t<single_boundary_enabled>>
	constexpr dimension(const dimension<old_lower, old_upper>& other,
			const lower_boundary_type& lower) :
		dimension(other, lower, lower) {}
private:
	const lower_boundary_type _lower;
	const upper_boundary_type _upper;

	constexpr dimension(units::length size, const lower_boundary_type& lower,
			const upper_boundary_type& upper, unsigned id) :
		dimension_base(size, id), _lower(lower), _upper(upper) {}
};

// Deduction help for single-parameter-type boundary
template <typename boundary_type>
dimension(units::length, const boundary_type&) -> dimension<boundary_type, boundary_type>;

template <typename old_lower, typename old_upper, typename boundary_type>
dimension(const dimension<old_lower, old_upper>&, const boundary_type&) ->
	dimension<boundary_type, boundary_type>;

} // namespace __1

using __1::dimension;

template <typename> struct is_dimension : std::false_type {};
template <typename lower, typename upper>
struct is_dimension<__1::dimension<lower, upper>> : std::true_type {};

template <typename dimension_type>
inline constexpr auto is_dimension_v = is_dimension<dimension_type>::value;

} // namespace fd
