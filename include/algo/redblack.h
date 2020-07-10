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

// k-colors a regular grid (smallest allowed k determined by connectivity, but
// typically k=2 or 3).
//
//   r--b--r--b--r--b
//   |  |  |  |  |  |
//   b--r--b--r--b--r
//   |  |  |  |  |  |     +----------+
//   r--b--r--b--r--b     | Legend   |
//   |  |  |  |  |  |     | r: red   |
//   b--r--b--r--b--r     | b: black |
//   |  |  |  |  |  |     +----------+
//   r--b--r--b--r--b
//   |  |  |  |  |  |
//   b--r--b--r--b--r
//
// Then permutes data on the grid so that values at nodes colored 0 are listed
// consecutively, followed by 1s, 2s, ..., ks.
//
//   E--F--G--H--I--J
//   |  |  |  |  |  |
//   y--z--A--B--C--D
//   |  |  |  |  |  |
//   s--t--u--v--w--x      a c e h j l m o q t v x
//   |  |  |  |  |  |  ->  y A C F H J b d f g i k
//   m--n--o--p--q--r      n p r s u w z B D E G I
//   |  |  |  |  |  |
//   g--h--i--j--k--l
//   |  |  |  |  |  |
//   a--b--c--d--e--f
//
// Used by symmetric incomplete LU factorization, Gauss-Seidel iteration

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

template <typename grid_type>
constexpr auto
colors(const grid_type& grid)
{
	const auto& components = grid.components();
	return get_colors(components);
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

template <typename grid_type>
class permuter {
private:
	static constexpr auto dimensions = grid_type::dimensions;
	using sequence = std::make_index_sequence<dimensions>;
	int colors;
	int sizes[dimensions];
protected:
	template <std::size_t ... ns>
	permuter(std::index_sequence<ns...>, const grid_type& grid) :
		colors(impl::colors(grid)),
		sizes{std::get<ns>(fd::sizes(grid))...} {}

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

	permuter(const grid_type& grid) :
		permuter(sequence(), grid) {}
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
	permute(matrix m) const
	{
		auto* pdata = permutation.data();
		auto* qdata = inverse.data();
		return permute_with(std::move(m), pdata, qdata);
	}

	virtual matrix
	unpermute(matrix m) const
	{
		auto* pdata = inverse.data();
		auto* qdata = permutation.data();
		return permute_with(std::move(m), pdata, qdata);
	}

	virtual vector
	permute(vector v) const
	{
		auto* pdata = inverse.data();
		return permute_with(std::move(v), pdata);
	}

	virtual vector
	unpermute(vector v) const
	{
		auto* pdata = permutation.data();
		return permute_with(std::move(v), pdata);
	}

	template <typename grid_type>
	redblack(const grid_type& grid) :
		coloring{fd::size(grid), impl::colors(grid)},
		permutation(coloring::size()), inverse(coloring::size())
	{
		impl::permuter permute{grid};
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
