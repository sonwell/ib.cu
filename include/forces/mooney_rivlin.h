#pragma once
#include "bases/geometry.h"
#include "types.h"
#include "load.h"

namespace forces {

struct mooneyrivlin {
	struct helper {
		double e, f, g;
		double eu, ev, fu, fv, gu, gv;
		double i, iu, iv;

		constexpr helper(const info<2>& load) :
			e(algo::dot(load.t[0], load.t[0])),
			f(algo::dot(load.t[0], load.t[1])),
			g(algo::dot(load.t[1], load.t[1])),
			eu(2 * algo::dot(load.t[0], load.tt[0])),
			ev(2 * algo::dot(load.t[0], load.tt[1])),
			fu(algo::dot(load.t[0], load.tt[1]) + algo::dot(load.tt[0], load.t[1])),
			fv(algo::dot(load.t[0], load.tt[2]) + algo::dot(load.tt[1], load.t[1])),
			gu(2 * algo::dot(load.t[1], load.tt[1])),
			gv(2 * algo::dot(load.t[1], load.tt[2])),
			i(e * g - f * f),
			iu(eu * g + e * gu - 2 * f * fu),
			iv(ev * g + e * gv - 2 * f * fv) {}
	};

	double shear;
	double bulk;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using bases::reference;
		using bases::current;
		const auto& orig = object.geometry(reference).sample;
		const auto& curr = object.geometry(current).sample;
		loader original{orig};
		loader deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=, shear=shear, bulk=bulk] __device__ (int tid)
		{
			auto orig = original(tid);
			auto curr = deformed(tid);
			auto [oe, of, og, oeu, oev, ofu, ofv,
				 ogu, ogv, odetg, odetgu, odetgv] = helper{orig};
			auto [ce, cf, cg, ceu, cev, cfu, cfv,
				 cgu, cgv, cdetg, cdetgu, cdetgv] = helper{curr};

			auto detfi = 1 / sqrt(odetg);
			auto detfiu = - odetgu * detfi / (2 * odetg);
			auto detfiv = - odetgv * detfi / (2 * odetg);

			auto c0 = shear * detfi;
			auto c0u = shear * detfiu;
			auto c0v = shear * detfiv;

			auto c1 = bulk * detfi;
			auto c1u = bulk * detfiu;
			auto c1v = bulk * detfiv;

			auto e = c0 * og + c1 * cg;
			auto eu = c0u * og + c0 * ogu + c1u * cg + c1 * cgu;
			auto f = c0 * of + c1 * cf;
			auto fu = c0u * of + c0 * ofu + c1u * cf + c1 * cfu;
			auto fv = c0v * of + c0 * ofv + c1v * cf + c1 * cfv;
			auto g = c0 * oe + c1 * ce;
			auto gv = c0v * oe + c0 * oev + c1v * ce + c1 * cev;

			auto& [u, v] = curr.t;
			auto& [uu, uv, vv] = curr.tt;

			for (int i = 0; i < 3; ++i)
				fdata[n * i + tid] = orig.s * detfi * (
						+ (e * uu[i] + eu * u[i])
						- (2 * f * uv[i] + fu * v[i] + fv * u[i])
						+ (g * vv[i] + gv * v[i]));
		};
		util::transform(k, n);
		return f;
	}

	constexpr mooneyrivlin(double shear, double bulk) :
		shear(shear), bulk(bulk) {}
};

} // namespace forces