#pragma once
#include <array>
#include "util/functional.h"
#include "bases/geometry.h"
#include "types.h"

namespace forces {

template <int dims>
struct info {
	static constexpr auto dimensions = dims;
	static constexpr auto nsd = (dims * (dims+1)) / 2;
	using vector_type = std::array<double, dimensions+1>;
	vector_type x;
	std::array<vector_type, dimensions> t;
	vector_type n;
	std::array<vector_type, nsd> tt;
	double s;
};

template <int dims>
class loader {
public:
	static constexpr auto dimensions = dims;

	constexpr auto size() const { return sz; }

	constexpr auto
	operator[](int tid) const
	{
		info<dimensions> load;
		auto n = sz.rows * sz.cols;
		auto j = tid % n;

		for (int i = 0; i < dimensions+1; ++i) {
			auto l = n * i + j;
			load.x[i] = xdata[l];
			for (int k = 0; k < dimensions; ++k)
				load.t[k][i] = tdata[k][l];
			load.n[i] = ndata[l];
			for (int k = 0; k < nsd; ++k)
				load.tt[k][i] = ttdata[k][l];
		}
		load.s = sdata[j];
		return load;
	}

	loader(const bases::geometry<dims>& g) :
		sz{g.position.rows(), g.position.cols() / (dimensions+1)},
		xdata(g.position.values()),
		tdata(pointer_array(g.tangents)),
		ndata(g.normal.values()),
		ttdata(pointer_array(g.second_derivatives)),
		sdata(g.sigma.values()) {}
private:
	static constexpr auto nsd = (dims * (dims+1)) / 2;

	template <typename tuple_type>
	static decltype(auto)
	pointer_array(const tuple_type& tuple)
	{
		using namespace util::functional;
		static constexpr auto size = std::tuple_size_v<tuple_type>;
		std::array<double*, size> r;
		auto k = [] (double*& p, const matrix& m) { p = m.values(); };
		map(k, r, tuple);
		return r;
	}

	linalg::size sz;
	double* xdata;
	std::array<double*, dimensions> tdata;
	double* ndata;
	std::array<double*, nsd> ttdata;
	double* sdata;
};

}
