#pragma once
#include "cublas/handle.h"
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct first_fundamental_form {
	double e, f, g, i;

	constexpr first_fundamental_form(const info<2>& load) :
		e(algo::dot(load.t[0], load.t[0])),
		f(algo::dot(load.t[0], load.t[1])),
		g(algo::dot(load.t[1], load.t[1])),
		i(e * g - f * f) {}
};

struct second_fundamental_form {
	double l, m, n, ii;

	constexpr second_fundamental_form(const info<2>& load) :
		l(algo::dot(load.tt[0], load.n)),
		m(algo::dot(load.tt[1], load.n)),
		n(algo::dot(load.tt[2], load.n)),
		ii(l * n - m * m) {}
};

struct preferential_bending {
	units::energy modulus;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		auto& origs = object.geometry(bases::reference).sample;
		auto& currs = object.geometry(bases::current).sample;
		loader loadc{currs};
		loader loadr{origs};

		matrix f{linalg::size(currs.position)};
		auto* fdata = f.values();
		auto n = f.rows() * f.cols() / 3;

		auto k = [=, modulus=(double) modulus] __device__ (int tid)
		{
			auto m = [] (const auto& geom)
			{
				auto [e, f, g, i] = first_fundamental_form{geom};
				auto [l, m, n, ii] = second_fundamental_form{geom};
				auto h = (l * g + n * e - 2 * f * m) / (2 * i);
				auto k = ii / i;
				return h * (h * h - k);
			};

			auto cg = loadc[tid];
			auto rg = loadr[tid];
			auto mag = -cg.s * modulus * (m(cg) - m(rg));
			for (int i = 0; i < 3; ++i)
				fdata[n * i + tid] = mag * cg.n[i];
		};
		util::transform(k, n);
		return f;
	}

	constexpr preferential_bending(units::energy modulus) :
		modulus(modulus) {}
};

} // namespace forces
