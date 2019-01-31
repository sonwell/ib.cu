#pragma once
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include "boundary.h"
#include "lwps/types.h"
#include "util/counter.h"

namespace fd {
	namespace dimension_impl {
		class base_dimension {
		private:
			// a bit of misdirection for the constexpr counter
			// see note in include/util/counter.h
			struct counter_base { static constexpr util::counter value{0u}; };
			template <unsigned> struct counter : counter_base { using counter_base::value; };

			const unsigned _id;
			const double _size;

		public:
			constexpr auto id() const { return _id; }
			constexpr auto size() const { return _size; }
			constexpr bool operator==(const base_dimension& other) const
				{ return _id == other._id; }
			constexpr bool operator!=(const base_dimension& other) const
				{ return _id != other._id; }
		protected:
			template <unsigned n = 0, unsigned id = next(counter<n>::value)>
			constexpr base_dimension(double size) :
				_id(id), _size(size) {}
		};

		template <typename lower_bdy, typename upper_bdy>
		class dimension : public base_dimension {
		static_assert(boundary::is_valid_combination_v<lower_bdy, upper_bdy>,
					"A dimension must either be periodic at both ends or neither end.");
		public:
			using lower_boundary_type = lower_bdy;
			using upper_boundary_type = upper_bdy;
		private:
			const lower_boundary_type _lower;
			const upper_boundary_type _upper;
		public:
			static constexpr bool solid_boundary = !std::is_same<
				lower_boundary_type, boundary::periodic>::value;

			const lower_boundary_type& lower() const { return _lower; }
			const upper_boundary_type& upper() const { return _upper; }

			constexpr dimension(double size, const lower_boundary_type& lower,
					const upper_boundary_type& upper) :
				base_dimension(size), _lower(lower), _upper(upper) {}

			template <typename OldLower, typename OldUpper>
			constexpr dimension(const dimension<OldLower, OldUpper>& other,
					const lower_boundary_type& lower,
					const upper_boundary_type& upper) :
				base_dimension(other), _lower(lower), _upper(upper) {}
		};

		template <typename boundary_type>
		class dimension<boundary_type, boundary_type> : public base_dimension {
			// The following message should never show, but we want the
			// side-effects of instantiating the template.
			static_assert(boundary::is_valid_combination_v<boundary_type, boundary_type>);
		private:
			const boundary_type _boundary;
		public:
			using lower_boundary_type = boundary_type;
			using upper_boundary_type = boundary_type;
			static constexpr bool solid_boundary = !std::is_same<
				boundary_type, boundary::periodic>::value;

			const boundary_type& lower() const { return _boundary; }
			const boundary_type& upper() const { return _boundary; }

			constexpr dimension(double size, const boundary_type& boundary) :
				base_dimension(size), _boundary(boundary) {}

			template <typename OldLower, typename OldUpper>
			constexpr dimension(const dimension<OldLower, OldUpper>& other,
					const boundary_type& boundary) :
				base_dimension(other), _boundary(boundary) {}
		};

		// Deduction help for single-parameter-type boundary
		template <typename boundary_type>
		dimension(double, const boundary_type&) -> dimension<boundary_type, boundary_type>;

		struct bad_grid_points : std::runtime_error
		{
			bad_grid_points() : std::runtime_error(
					"Non-integer number of grid points in dimension. Make sure "
					"the size of the dimension * the resolution is an integer.") {}
		};

		template <typename lower_bdy, typename upper_bdy>
		class view : public dimension<lower_bdy, upper_bdy> {
		private:
			const lwps::index_type _resolution;
		public:
			using dimension_type = dimension<lower_bdy, upper_bdy>;
			using typename dimension_type::lower_boundary_type;
			using typename dimension_type::upper_boundary_type;

			using dimension_type::size;
			using dimension_type::lower;
			using dimension_type::upper;

			constexpr lwps::index_type gridpts() const { return size() * _resolution; }
			constexpr const lwps::index_type& resolution() const { return _resolution; }

			constexpr view(lwps::index_type resolution, const dimension_type& dimension) :
				dimension_type(dimension), _resolution(resolution)
			{
				double grid_points = dimension.size() * resolution;
				if (grid_points < 0 || grid_points - ((lwps::index_type) grid_points) != 0.0)
					throw bad_grid_points();
			}
		};
	}

	using dimension_impl::dimension;
}
