#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct tension {
private:
	using tangents = tangents<2>;
	using seconds = seconds<2>;
	using metric = metric<2>;
	using christoffel = christoffel<2>;

	struct load : loader<tangents>, loader<seconds>, loader<measure> {
		struct payload : tangents, seconds, measure {
			using tangents::t;
			using seconds::tt;
			using measure::s;
		};

		constexpr payload
		operator[](int i) const
		{
			return {{loader<tangents>::operator[](i)},
			        {loader<seconds>::operator[](i)},
			        {loader<measure>::operator[](i)}};
		}

		load(const bases::geometry<2>& g) :
			loader<tangents>(g),
			loader<seconds>(g),
			loader<measure>(g) {}
	};

	struct helper {
		double e, f, g;
		double eu, fu, gu, ev, fv, gv;
		double i, iu, iv;

		constexpr helper(const metric& m, const christoffel& c) :
			e(m[0][0]),         f(m[0][1]),                  g(m[1][1]),
			// g_{ij,k} = Γ_{ik,j} + Γ{kj,i}, Γ_{ij,k} = Γ_{ji,k}
			eu(2 * c[0][0][0]), fu(c[0][0][1] + c[0][1][0]), gu(2 * c[1][0][1]),
			ev(2 * c[0][1][0]), fv(c[0][1][1] + c[1][1][0]), gv(2 * c[1][1][1]),
			i(e * g - f * f),
			iu(eu * g + e * gu - 2 * f * fu),
			iv(ev * g + e * gv - 2 * f * fv) {}

		constexpr helper(const load::payload& pt) :
			helper((const metric&) pt, (const christoffel&) pt) {}
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
		load original{orig};
		load deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__ (int tid, auto w)
		{
			auto orig = original[tid];
			auto curr = deformed[tid];
			auto [oe, of, og, oeu, ofu, ogu, oev, ofv, ogv, oi, oiu, oiv] = helper{orig};
			auto [ce, cf, cg, ceu, cfu, cgu, cev, cfv, cgv, ci, ciu, civ] = helper{curr};

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
	using tangents = tangents<1>;
	using seconds = seconds<1>;
	using metric = metric<1>;
	using christoffel = christoffel<1>;

	struct load : loader<tangents>, loader<seconds>, loader<measure> {
		struct payload : tangents, seconds, measure {
			using tangents::t;
			using seconds::tt;
			using measure::s;
		};

		constexpr payload
		operator[](int i) const
		{
			return {{loader<tangents>::operator[](i)},
			        {loader<seconds>::operator[](i)},
			        {loader<measure>::operator[](i)}};
		}

		load(const bases::geometry<1>& g) :
			loader<tangents>(g),
			loader<seconds>(g),
			loader<measure>(g) {}
	};

	struct helper {
		double i, iu;

		constexpr helper(const metric& m, const christoffel& c) :
			i(m[0][0]), iu(2 * c[0][0][0]) {}

		constexpr helper(const load::payload& pt) :
			helper((const metric&) pt, (const christoffel&) pt) {}
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
		load original{orig};
		load deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 2;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=] __device__ (int tid, auto w)
		{
			auto orig = original[tid];
			auto curr = deformed[tid];
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

			for (int j = 0; j < 2; ++j)
				fdata[n * j + tid] = orig.s / oi * (
						+ phi * uu[j]
						- odi / (2 * oi) * phi * u[j]
						+ phiu * u[j]
				);
		};
		util::transform(k, n, w);
		return f;
	}

	constexpr tension1d() = default;
};

} // namespace forces
