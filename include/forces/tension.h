#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct tension {
private:
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
protected:
	template <typename object_type, typename constitutive_law_type>
	decltype(auto)
	operator()(const object_type& object, constitutive_law_type&& w) const
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
		auto k = [=] __device__ (int tid, auto w)
		{
			auto orig = original(tid);
			auto curr = deformed(tid);
			auto [oe, of, og, oeu, oev, ofu, ofv, ogu, ogv, oi, oiu, oiv] = helper{orig};
			auto [ce, cf, cg, ceu, cev, cfu, cfv, cgu, cgv, ci, ciu, civ] = helper{curr};

			// invariants
			auto i1 = (ce * og + oe * cg - 2 * of * cf) / oi;
			auto i2 = ci / oi;

			auto i1u = (ceu * og + ce * ogu + oeu * cg + oe * cgu - 2 * (ofu * cf + of * cfu) - i1 * oiu) / oi;
			auto i1v = (cev * og + ce * ogv + oev * cg + oe * cgv - 2 * (ofv * cf + of * cfv) - i1 * oiv) / oi;
			auto i2u = (ciu - i2 * oiu) / oi;
			auto i2v = (civ - i2 * oiv) / oi;

			// material functions
			auto [w1, w2, w11, w12, w22] = w(i1, i2);

			auto w1u = w11 * i1u + w12 * i2u;
			auto w1v = w11 * i1v + w12 * i2v;
			auto w2u = w12 * i1u + w22 * i2u;
			auto w2v = w12 * i1v + w22 * i2v;

			auto& [u, v] = curr.t;
			auto& [uu, uv, vv] = curr.tt;

			for (int i = 0; i < 3; ++i)
				fdata[n * i + tid] = orig.s / oi * (
						+     (w1 * og + w2 * cg) * uu[i]
						- 2 * (w1 * of + w2 * cf) * uv[i]
						+     (w1 * oe + w2 * ce) * vv[i]
						- (w1 * og + w2 * cg) * oiu / (2 * oi) * u[i]
						+ (w1 * of + w2 * cf) * oiv / (2 * oi) * u[i]
						+ (w1 * of + w2 * cf) * oiu / (2 * oi) * v[i]
						- (w1 * oe + w2 * ce) * oiv / (2 * oi) * v[i]
						+ (w1u * og + w1 * ogu + w2u * cg + w2 * cgu) * u[i]
						- (w1v * of + w1 * ofv + w2v * cf + w2 * cfv) * u[i]
						- (w1u * of + w1 * ofu + w2u * cf + w2 * cfu) * v[i]
						+ (w1v * oe + w1 * oev + w2v * ce + w2 * cev) * v[i]
				);
		};
		util::transform(k, n, w);
		return f;
	}

	constexpr tension() = default;
};

struct tension1d {
private:
	struct helper {
		double i, di;

		constexpr helper(const info<1>& load) :
			i(algo::dot(load.t[0], load.t[0])),
			di(2 * algo::dot(load.t[0], load.tt[0])) {}
	};
protected:
	template <typename object_type, typename constitutive_law_type>
	decltype(auto)
	operator()(const object_type& object, constitutive_law_type&& w) const
	{
		using bases::reference;
		using bases::current;
		const auto& orig = object.geometry(reference).sample;
		const auto& curr = object.geometry(current).sample;
		loader original{orig};
		loader deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 2;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__ (int tid, auto w)
		{
			auto orig = original(tid);
			auto curr = deformed(tid);
			auto [oi, odi] = helper{orig};
			auto [ci, cdi] = helper{curr};

			// invariants
			auto i = ci / oi;
			auto iu = (cdi - i * odi) / oi;

			// material functions
			auto [phi, phi1] = w(i);

			auto phiu = phi1 * iu;

			auto& [u] = curr.t;
			auto& [uu] = curr.tt;
			auto& nrml = curr.n;

			for (int j = 0; j < 2; ++j)
				fdata[n * j + tid] = orig.s / oi * (
						+ (phi / i) * uu[j]
						- 0.5 * iu / i * phi * u[j]
						+ phiu * u[j]
				);
		};
		util::transform(k, n, w);
		return f;
	}

	constexpr tension1d() = default;
};

} // namespace forces
