#pragma once
#include <array>
#include <functional>

#include "util/launch.h"
#include "util/functional.h"
#include "util/memory.h"
#include "fd/domain.h"
#include "fd/size.h"
#include "cuda/copy.h" // XXX
#include "types.h"
#include "coloring.h"

namespace algo {
namespace impl {

using result_type = std::array<int, 2>;

template <typename views_type>
constexpr int
get_colors(const views_type& views, int colors = 2)
{
	using namespace util::functional;
	auto all = partial(foldl, std::logical_and<bool>(), true);
	auto acceptable = [&] (const auto& view)
	{
		auto cells = view.cells();
		return view.solid_boundary || cells <= 1
			|| (cells - 1) % colors != 0;
	};
	return apply(all, map(acceptable, views)) ?
		colors : get_colors(views, colors+1);
}

template <typename domain_type>
constexpr auto
colors(const domain_type& domain)
{
	auto&& views = fd::dimensions(domain);
	return get_colors(views);
}

struct level {
	int* sums;
	int total;
	const int size;

	template <typename func_type>
	__host__ __device__ void
	update(func_type f)
	{
		for (int i = 0; i < size; ++i)
			sums[i * size + 1] = f(i);
		int ctotal;
		for (int j = 0; j < size; ++j) {
			ctotal = sums[j * size + 1];
			for (int i = 2; i < size; ++i) {
				ctotal += sums[((j + size + 1 - i) % size) * size + 1];
				sums[j * size + i] = ctotal;
			}
		}
		total = ctotal + sums[1];
	}

	__host__ __device__ level(int size) :
		sums(new int[size * size]{0}),
		total(0), size(size) {}
	__host__ __device__ ~level() { delete[] sums; }
};

__host__ __device__ result_type
recurse(int i, int colors, struct level& lev)
{
	return {0, 0};
}

template <typename ... arg_types>
__host__ __device__ result_type
recurse(int i, int colors, struct level& lev, int size, arg_types ... args)
{
	int index = i % size;
	int q = size / colors, r = size % colors;
	int quot = index / colors, rem = index % colors;

	if constexpr(sizeof...(arg_types) == 0) {
		lev.update([&] (int i) { return q + (i < r); });
		return {quot, rem};
	} else {
		auto [p, c] = recurse(i / size, colors, lev, args...);
		int color = (c + rem) % colors;
		int result = p + lev.sums[color * colors + rem] + quot * lev.total;
		lev.update([&] (int i) { return q * lev.total + lev.sums[i * colors + r]; });
		return {result, color};
	}
}

template <typename domain_type>
class permuter {
private:
	static constexpr auto dimensions = domain_type::ndim;
	using sequence = std::make_index_sequence<dimensions>;
	int colors;
	int sizes[dimensions];
protected:
	template <std::size_t ... ns, typename grid_dim>
	permuter(std::index_sequence<ns...>, const domain_type& domain,
			const grid_dim& dim) :
		colors(impl::colors(domain)),
		sizes{std::get<ns>(fd::sizes(domain, dim))...} {}

	template <std::size_t ... ns>
	__host__ __device__ int
	entry(std::index_sequence<ns...>, int tid, int* starts) const
	{
		level lev(colors);
		auto [result, color] = recurse(tid, colors, lev, sizes[ns]...);
		auto index = (color + colors - 1) % colors;
		auto offset = lev.sums[index * colors + color];
		if (!result) starts[color] = offset;
		return result + offset;
	}
public:
	__host__ __device__ int
	operator()(int tid, int* starts) const
	{ return entry(sequence(), tid, starts); }

	template <typename grid_dim>
	permuter(const domain_type& domain, const grid_dim& dim) :
		permuter(sequence(), domain, dim) {}
};

} // namespace impl

class redblack : public coloring {
private:
	util::memory<int> permutation;
	util::memory<int> inverse;
protected:
	using coloring::permute_with;
public:
	virtual matrix
	permute(const matrix& m) const
	{
		auto* pdata = permutation.data();
		auto* qdata = inverse.data();
		return permute_with(m, pdata, qdata);
	}

	virtual matrix
	unpermute(const matrix& m) const
	{
		auto* pdata = inverse.data();
		auto* qdata = permutation.data();
		return permute_with(m, pdata, qdata);
	}

	virtual vector
	permute(const vector& v) const
	{
		auto* pdata = inverse.data();
		return permute_with(v, pdata);
	}

	virtual vector
	unpermute(const vector& v) const
	{
		auto* pdata = permutation.data();
		return permute_with(v, pdata);
	}

	template <typename domain_type, typename grid_dim>
	redblack(const domain_type& domain, const grid_dim& dim) :
		coloring{fd::size(domain, dim), impl::colors(domain)},
		permutation(coloring::size()), inverse(coloring::size())
	{
		impl::permuter permute{domain, dim};
		auto colors = coloring::colors();
		auto size = coloring::size();
		util::memory<int> starts(colors+1);

		auto* pdata = permutation.data();
		auto* qdata = inverse.data();
		auto* sdata = starts.data();
		auto k = [=] __device__ (int tid)
		{
			auto j = permute(tid, sdata);
			pdata[j] = tid;
			qdata[tid] = j;
			if (!tid) sdata[colors] = size;
		};
		util::transform<128, 7>(k, size);
		cuda::dtoh(coloring::starts(), sdata, colors+1);
	}
};

} // namespace algo
