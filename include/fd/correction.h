#pragma once
#include <cstddef>
#include "util/array.h"
#include "util/launch.h"
#include "types.h"
#include "grid.h"
#include "boundary.h"
#include "dimension.h"
#include "discretization.h"

namespace fd {
namespace boundary {
namespace correction {

template <std::size_t N> struct order :
	std::integral_constant<std::size_t, N> {};
inline constexpr order<0> zeroth_order;
inline constexpr order<1> first_order;
inline constexpr order<2> second_order;

} // namespace correction

namespace impl {

inline matrix
single_entry(int rows, int cols, int row, int col, double value)
{
	if (!rows || !cols || !value)
		return matrix{rows, cols};

	matrix r{rows, cols, 1, rows+1, 1, 1};
	auto* sdata = r.starts();
	auto* idata = r.indices();
	auto* vdata = r.values();

	auto k = [=] __device__ (int tid)
	{
		sdata[tid] = row < tid;

		if (!tid) {
			sdata[rows] = 1;
			idata[0] = col + indexing_base;
			vdata[0] = value;
		}
	};
	util::transform<128, 4>(k, rows);
	return r;
}

} // namespace impl

template <typename collocation, typename view_type>
class corrector : public discretization<collocation, view_type> {
private:
	using base = discretization<collocation, view_type>;
	using lower_tag = impl::lower_tag;
	using upper_tag = impl::upper_tag;
public:
	using base::solid_boundary;
	using typename base::params;
	using base::interior;
	using base::boundary;
	using base::identity;

	matrix interior(int, int, double, lower_tag) const;
	matrix interior(int, int, double, upper_tag) const;
	matrix boundary(int, int, double, lower_tag) const;
	matrix boundary(int, int, double, upper_tag) const;

	template <std::size_t N> constexpr params
		identity(const correction::order<N>&) const;

	constexpr corrector(const view_type& view) : base(view) {}
};

template <typename collocation, typename view_type>
template <std::size_t N>
constexpr auto
corrector<collocation, view_type>::
identity(const correction::order<N>&) const -> params
{
	if constexpr (N == 2)
		return base::identity();
	else
		return {0, 0};
}

template <typename collocation, typename view_type>
inline matrix
corrector<collocation, view_type>::
interior(int rows, int cols, double scale, lower_tag) const
{
	using impl::single_entry;
	if constexpr (solid_boundary)
		return single_entry(rows, cols, 0, 0, base::interior().lower * scale);
	return single_entry(rows, cols, 0, cols-1, scale);
}

template <typename collocation, typename view_type>
inline matrix
corrector<collocation, view_type>::
interior(int rows, int cols, double scale, upper_tag) const
{
	using impl::single_entry;
	if constexpr (solid_boundary)
		return single_entry(rows, cols, rows-1, cols-1, base::interior().upper * scale);
	return single_entry(rows, cols, rows-1, 0, scale);
}

template <typename collocation, typename view_type>
inline matrix
corrector<collocation, view_type>::
boundary(int rows, int cols, double scale, lower_tag) const
{
	using impl::single_entry;
	if constexpr (solid_boundary)
		return single_entry(rows, cols, 0, 0, base::boundary().lower * scale);
	return matrix{rows, cols};
}

template <typename collocation, typename view_type>
inline matrix
corrector<collocation, view_type>::
boundary(int rows, int cols, double scale, upper_tag) const
{
	using impl::single_entry;
	if constexpr (solid_boundary)
		return single_entry(rows, cols, rows-1, cols-1, base::boundary().upper * scale);
	return matrix{rows, cols};
}

} // namespace boundary
} // namespace fd
