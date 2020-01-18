#pragma once
#include <thrust/execution_policy.h>
#include "cuda/event.h"
#include "util/log.h"
#include "util/functional.h"
#include "util/iterators.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "delta.h"
#include "roma.h"
#include "types.h"
#include "sweep.h"
#include "indexing.h"

namespace ib {

struct timer {
	std::string id;
	cuda::event start, stop;

	timer(std::string id) :
		id(id) { start.record(); }
	~timer() {
		stop.record();
		util::logging::info(id, ": ", stop-start, "ms");
	}
};

namespace interpolation {

struct clamped {
	static constexpr auto
	clamp(int index, int lower, int upper)
	{
		return index < lower ? lower :
			index >= upper ? upper - 1 : index;
	}

	int index, lower, upper;

	constexpr operator int() const { return index; }

	constexpr clamped(int index, int lower, int upper) :
		index(clamp(index, lower, upper)),
		lower(lower), upper(upper) {}
};

constexpr auto
combine(const clamped& l, const clamped& r)
{
	auto weight = l.upper - l.lower;
	auto index = l.index + weight * r.index;
	auto lower = weight * r.lower;
	auto upper = weight * r.upper;
	return clamped{index, lower, upper};
}

template <typename grid_type>
struct sorter : ib::indexing::sorter<grid_type> {
	using sort_index_type = clamped;
	using ib::indexing::sorter<grid_type>::sorter;
};

} // namespace interpolation

template <typename grid_tag, typename domain_type>
struct interpolate {
public:
	static constexpr auto dimensions = domain_type::dimensions;
private:
	static constexpr thrust::device_execution_policy<thrust::system::cuda::tag> exec = {};
	static constexpr ib::delta::roma phi;
	using traits = ib::delta::traits<ib::delta::roma>;
	static constexpr auto values = ib::detail::cpow(traits::meshwidths, dimensions);
	static constexpr auto per_sweep = values;
	static constexpr auto sweeps = (values + per_sweep - 1) / per_sweep;

	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp) { return fd::grid{tag, domain, comp}; };
		return map(k, fd::components(domain));
	}

	template <typename grid_type>
	static auto
	accumulate(int n, const grid_type& grid, double* vdata, const matrix& x, const vector& u)
	{
		using point_type = typename grid_type::point_type;
		using sorter = interpolation::sorter<grid_type>;

		auto* xdata = x.values();
		auto* udata = u.values();

		ib::indexer idx{sorter{grid}};
		auto k = [=] __device__(int tid)
		{
			double v = 0.0;
			point_type z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto j = idx.sort(z);

			for (int i = 0; i < sweeps; ++i) {
				ib::sweep sweep{i, per_sweep, phi, idx};
				auto w = sweep.values(z);
				auto k = sweep.indices(j);
				for (auto [k, w]: util::zip(k, w))
					if (k >= 0) v += w * udata[k];
			}
			vdata[tid] = v;
		};
		util::transform(k, n);
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(), std::declval<domain_type>()));
	grids_type grids;
public:
	constexpr interpolate(const grid_tag& tag, const domain_type& domain) :
		grids(construct(tag, domain)) {}

	template <typename tuple_type>
	auto
	operator()(int n, const matrix& x, const tuple_type& u) const
	{
		timer s{"interpolate"};
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;

		matrix v{linalg::size(x), linalg::zero};
		auto* vdata = v.values();
		auto k = [&] (const auto& grid, const vector& u, auto m)
		{
			static constexpr auto i = decltype(m)::value;
			auto* v = &vdata[n * i];
			return accumulate(n, grid, v, x, u);
		};
		map(k, grids, u, sequence{});
		return v;
	}
};

} // namespace ib
