#pragma once
#include <thrust/execution_policy.h>
#include "util/functional.h"
#include "util/iterators.h"
#include "util/ranges.h"
#include "fd/domain.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "fd/grid.h"
#include "types.h"
#include "sweep.h"
#include "delta.h"
#include "roma.h"
#include "cosine.h"
#include "interpolate.h"

namespace ib {
namespace novel {

template <typename grid_tag, typename domain_type, typename delta_type>
struct spread {
private:
	using cuda_tag = thrust::system::cuda::tag;
	using policy = thrust::device_execution_policy<cuda_tag>;
	using traits = delta::traits<delta_type>;
	static constexpr policy exec;
	static constexpr auto dimensions = domain_type::dimensions;
	static constexpr auto meshwidths = traits::meshwidths;
	static constexpr auto values = detail::cpow(meshwidths, dimensions);
	static constexpr delta_type phi;
	using point = ib::point<dimensions>;

	static constexpr auto
	largest_divisor_under(int n)
	{
		if (n >= values) return values;
		for (int i = n; i > 0; --i)
			if (values % i == 0)
				return i;
		return 1;
	}

	static constexpr auto per_sweep = largest_divisor_under(10);
	using container_type = values_container<per_sweep>;

	static constexpr auto
	construct(const grid_tag& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto& comp) { return fd::grid{tag, domain, comp}; };
		return map(k, fd::components(domain));
	}

	static auto
	uniques(int n, const util::memory<int>& indices)
	{
		util::memory<int> buffer(n);
		auto* idata = indices.data();
		auto* bdata = buffer.data();
		return thrust::unique_copy(exec, idata, idata+n, bdata) - bdata;
	}

	template <typename grid_type>
	static auto
	index(int n, const grid_type& grid, const matrix& x,
			util::memory<int>& i, util::memory<int>& p)
	{
		auto* xdata = x.values();
		auto* idata = i.data();
		auto* jdata = p.data();

		indexing::sorter idx{grid};
		ib::sweep sweep{0, values, phi, idx};
		auto k = [=] __device__ (int tid)
		{
			point z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + tid];
			auto j = sweep.sort(z);
			idata[tid] = j;
			jdata[tid] = tid;
		};
		util::transform<128, 3>(k, n);
		thrust::sort_by_key(exec, idata, idata+n, jdata);
	}

	template <typename grid_type>
	static auto
	reduce(int n, int q, const grid_type& grid, const matrix& x, const double* fdata,
			vector& f, vector& buf, const util::memory<int>& i, const util::memory<int>& p)
	{
		using namespace util::functional;
		using namespace util::math;
		using util::ranges::enumerate;
		constexpr auto sweeps = (values + per_sweep - 1) / per_sweep;
		constexpr auto gr = [] (auto&& ... c) { return (c.resolution() * ... * 1); };

		auto res = apply(gr, grid.components());
		auto size = fd::size(grid);
		util::memory<container_type> values(n);
		util::memory<container_type> reduced_values(q);
		util::memory<int> reduced_keys(q);

		auto* vdata = values.data();
		auto* wdata = reduced_values.data();
		auto* kdata = reduced_keys.data();
		auto* idata = i.data();
		auto* jdata = p.data();
		auto* xdata = x.values();

		auto* odata = f.values();
		auto* bdata = buf.values();

		indexing::sorter idx{grid};
		auto k = [=] __device__ (int tid, int s)
		{
			sweep sweep{s, per_sweep, phi, idx};
			auto j = jdata[tid];
			double f = fdata[j];
			point z;
			for (int i = 0; i < dimensions; ++i)
				z[i] = xdata[n * i + j];
			auto w = sweep.values(z);
			container_type v;
			for (auto [i, w]: w | enumerate)
				v[i] = w * f;
			vdata[tid] = v;
		};

		auto l = [=] __device__ (int tid, int s)
		{
			sweep sweep{s, per_sweep, phi, idx};
			auto k = kdata[tid];
			auto v = wdata[tid];
			auto j = sweep.indices(k);
			for (auto [i, j]: j | enumerate)
				if (j >= 0) bdata[i * size + j] += res * v[i];
		};

		for (int i = 0; i < sweeps; ++i) {
			util::transform(k, n, i);
			thrust::reduce_by_key(exec, idata, idata+n, vdata, kdata, wdata);
			util::transform(l, q, i);
		}

		auto r = [=] __device__ (int tid)
		{
			double t = 0.0;
			for (int i = 0; i < per_sweep; ++i) {
				t += odata[i * size + tid];
				bdata[i * size + tid] = 0.0;
			}
			odata[tid] = t;
		};
		util::transform<128, 3>(r, size);
	}

	using grids_type = decltype(construct(std::declval<grid_tag>(),
	                                      std::declval<domain_type>()));

	static auto
	buffer_size(const grids_type& grids)
	{
		using namespace util::functional;
		constexpr auto max = partial(foldl, [] (auto l, auto r) { return l < r ? r : l; });
		constexpr auto size = [] (const auto& g) { return fd::size(g); };
		return per_sweep * apply(max, map(size, grids));
	}

	grids_type grids;
public:
	constexpr spread(const grid_tag& tag, const domain_type& domain, delta_type) :
		grids(construct(tag, domain)) {}

	template <typename tuple_type>
	void
	operator()(int n, const matrix& x, const matrix& fl, tuple_type& fe) const
	{
		using namespace util::functional;
		using sequence = std::make_index_sequence<dimensions>;
		util::memory<int> keys(n);
		util::memory<int> perm(n);
		vector buf{buffer_size(grids), linalg::zero};
		auto* fdata = fl.values();
		auto k = [&] (const auto& grid, vector& fe, auto m)
		{
			static constexpr auto i = decltype(m)::value;
			index(n, grid, x, keys, perm);
			auto unique = uniques(n, keys);
			reduce(n, unique, grid, x, &fdata[n * i],
			       fe, buf, keys, perm);
		};
		map(k, grids, fe, sequence{});
	}

	auto
	operator()(int n, const matrix& x, const matrix& fl) const
	{
		using namespace util::functional;
		auto s = [] (const auto& g) { return vector{fd::size(g), linalg::zero}; };
		auto fe = map(s, grids);
		operator()(n, x, fl, fe);
		return fe;
	}
};

using ib::interpolate;

} // namespace novel
} // namespace ib
