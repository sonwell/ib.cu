#pragma once
#include "util/launch.h"
#include "types.h"

namespace bases {

// Fills the Φ portion of a matrix of the form
//
//     [  Φ   P ]  or  [ Φ  P ],
//     [ P^T  0 ]
//
// which are values from (a derivative of) an RBF.
// NB: Derivatives of RBFs are not themselves RBFs, so we do not check that
// `rbf` is actually an RBF type.
template <int dimensions, typename rbf>
static auto
fill(const matrix& xs, const matrix& xd, rbf phi, matrix& r)
{
	static constexpr auto patch_rows = 16;
	static constexpr auto patch_cols = 16;
	static constexpr auto nt = patch_rows * patch_cols;
	static constexpr auto vt = 1;

	auto ns = xs.rows();
	auto nd = xd.rows();
	auto rows = r.rows();

	auto* sdata = xs.values();
	auto* ddata = xd.values();
	auto* rdata = r.values();

	auto row_patches = (ns + patch_rows - 1) / patch_rows;
	auto col_patches = (nd + patch_cols - 1) / patch_cols;
	auto patches = row_patches * col_patches;
	int num_ctas = (patches + vt - 1) / vt;

	auto k = [=] __device__ (int tid, int cta, auto f)
	{
		__shared__ struct {
			double s[dimensions * patch_rows];
			double d[dimensions * patch_cols];
		} shared;

		auto rel_row = tid % patch_rows;
		auto rel_col = tid / patch_rows;
		auto patch_row = (cta % row_patches) * patch_rows;
		auto patch_col = (cta / row_patches) * patch_cols;
		auto row = patch_row + rel_row;
		auto col = patch_col + rel_col;

		auto shared_row = patch_row + tid;
		auto shared_col = patch_col + tid;
		auto r_pred = shared_row < ns;
		auto c_pred = shared_col < nd;

		for (int i = 0; i < dimensions; ++i) {
			auto s = r_pred ? sdata[i * ns + shared_row] : 0;
			auto d = c_pred ? ddata[i * nd + shared_col] : 0;
			if (tid < patch_rows) shared.s[i * patch_rows + tid] = s;
			if (tid < patch_cols) shared.d[i * patch_cols + tid] = d;
		}
		__syncthreads();

		double sample[dimensions];
		double data[dimensions];
		for (int i = 0; i < dimensions; ++i) {
			sample[i] = shared.s[i * patch_rows + rel_row];
			data[i] = shared.d[i * patch_rows + rel_col];
		}

		auto value = f(sample, data);
		if (row < ns && col < nd)
			rdata[row + col * rows] = value;
	};
	util::launch<nt, vt>(k, num_ctas, phi);
}

// Construct a matrix of the form
//
//     [  Φ   P ]
//     [ P^T  0 ]
//
template <int dimensions, typename rbf, typename poly>
static auto
fill(const matrix& x, rbf phi, poly p)
{
	using params = double[dimensions];
	using poly_array = decltype(p(std::declval<params>()));
	static constexpr int np = std::tuple_size_v<poly_array>;
	auto nd = x.rows();
	auto rows = nd + np;
	matrix r{rows, rows};
	fill<dimensions>(x, x, phi, r);

	auto* ddata = x.values();
	auto* rdata = r.values();
	auto k = [=] __device__ (int tid)
	{
		params sample;
		for (int i = 0; i < dimensions; ++i)
			sample[i] = ddata[i * nd + tid];
		auto values = p(sample);
		for (int i = 0; i < np; ++i) {
			rdata[(nd + i) * rows + tid] = values[i];
			rdata[tid * rows + nd + i] = values[i];
		}
		if constexpr (np)
			for (int i = tid; i < np * np; i += nd) {
				auto row = i % np;
				auto col = i / np;
				rdata[(nd + col) * rows + nd + row] = 0;
			}
	};
	if constexpr (np)
		util::transform<128, 8>(k, nd);
	return r;
}

// Construct a matrix of the form
//
//     [ Φ  P ]
//
template <int dimensions, typename rbf, typename poly>
static auto
fill(const matrix& xs, const matrix& xd, rbf phi, poly p)
{
	using params = double[dimensions];
	using poly_array = decltype(p(std::declval<params>()));
	static constexpr int np = std::tuple_size_v<poly_array>;
	auto ns = xs.rows();
	auto nd = xd.rows();
	auto rows = ns;
	auto cols = nd + np;
	matrix r{rows, cols};
	fill<dimensions>(xs, xd, phi, r);

	auto* sdata = xs.values();
	auto* rdata = r.values();
	auto k = [=] __device__ (int tid)
	{
		params sample;
		for (int i = 0; i < dimensions; ++i)
			sample[i] = sdata[i * ns + tid];
		auto values = p(sample);
		for (int i = 0; i < np; ++i)
			rdata[(nd + i) * rows + tid] = values[i];
	};
	util::transform<128, 1>(k, ns);
	return r;
}

}
