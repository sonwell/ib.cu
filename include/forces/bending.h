#pragma once
#include "bases/geometry.h"
#include "bases/operators.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {
struct bending {
	units::energy modulus;
	bool preferred;

	// ds^2 = e da^1 + 2f da db + g db^2
	// Tells us stuff about lengths / areas
	struct first {
		double e, f, g, i;

		constexpr first(const info<2>& load) :
			e(algo::dot(load.t[0], load.t[0])),
			f(algo::dot(load.t[0], load.t[1])),
			g(algo::dot(load.t[1], load.t[1])),
			i(e * g - f * f) {}
	};

	// l da^2 + 2 m da db + n db^2
	// Tells us stuff about curvature
	struct second {
		double l, m, n, ii;

		constexpr second(const info<2>& load) :
			l(algo::dot(load.tt[0], load.n)),
			m(algo::dot(load.tt[1], load.n)),
			n(algo::dot(load.tt[2], load.n)),
			ii(l * n - m * m) {}
	};

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

	static decltype(auto)
	mean(const loader<2>& load)
	{
		auto [n, m] = load.size();
		matrix h{n, m};
		auto* hdata = h.values();
		auto k = [=] __device__ (int tid)
		{
			auto info = load[tid];
			auto [e, f, g, i] = first{info};
			auto [l, m, n, ii] = second{info};
			hdata[tid] = (l * g - 2 * m * f + n * e) / (2 * i);
		};
		util::transform<128, 3>(k, n*m);
		return h;
	}

	static decltype(auto)
	gaussian(const loader<2>& load)
	{
		auto [n, m] = load.size();
		matrix h{n, m};
		auto* hdata = h.values();
		auto k = [=] __device__ (int tid)
		{
			auto info = load[tid];
			auto [e, f, g, i] = first{info};
			auto [l, m, n, ii] = second{info};
			hdata[tid] = ii / i;
		};
		util::transform<128, 3>(k, n*m);
		return h;
	}

	decltype(auto)
	laplacian(matrix h, const loader<2>& load,
	          const bases::operators<2>& ops) const
	{
		auto c = solve(ops.restrictor, h);
		auto [n, m] = linalg::size(h);
		auto* hdata = h.values();

		std::array dh = {ops.first_derivatives[0] * c,
		                 ops.first_derivatives[1] * c};
		std::array d2h = {ops.second_derivatives[0] * c,
		                  ops.second_derivatives[1] * c,
		                  ops.second_derivatives[2] * c};
		auto* hu = dh[0].values();
		auto* hv = dh[1].values();
		auto* huu = d2h[0].values();
		auto* huv = d2h[1].values();
		auto* hvv = d2h[2].values();
		auto k = [=] __device__ (int tid)
		{
			auto info = load[tid];
			auto [e, f, g, eu, ev, fu, fv, gu, gv, i, iu, iv] = helper{info};
			auto lh = (g * huu[tid] - 2 * f * huv[tid] + e * hvv[tid] +
			           gu * hu[tid] - fu * hv[tid] - fv * hu[tid] + ev * hv[tid] +
			           iu * (g * hu[tid] - f * hv[tid]) +
			           iv * (e * hv[tid] - f * hu[tid])) / i;
			hdata[tid] = lh;
		};
		util::transform(k, n * m);
		return h;
	}

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		/* F_bend = -4κ(Δ(H-H₀) + 2(H-H₀)(H²-K))n̂ */
		loader curr{object.geometry(bases::current)};
		loader orig{object.geometry(bases::reference)};

		auto [n, m] = curr.size();
		matrix h{n, m};
		matrix r{n, m};
		auto* hdata = h.values();
		auto* rdata = r.values();
		auto k = [=, p=preferred] __device__ (int tid)
		{
			auto info = curr[tid];
			auto [e, f, g, i] = first{info};
			auto [l, m, n, ii] = second{info};
			auto h = (l * g - 2 * m * f + n * e) / (2 * i);
			auto k = ii / i;
			auto r = 2 * (h * h - k);
			if (p) {
				auto info = orig[tid];
				auto [e, f, g, i] = first{info};
				auto [l, m, n, ii] = second{info};
				h -= (l * g - 2 * m * f + n * e) / (2 * i);
			}
			hdata[tid] = h;
			rdata[tid] = h * r;
		};
		util::transform<128, 7>(k, n * m);
		auto dh = laplacian(std::move(h), curr, object.operators());

		matrix f{n, 3 * m};
		auto* fdata = f.values();
		auto* dhdata = dh.values();
		auto l = [=, n=n, m=m, k=modulus] __device__ (int tid)
		{
			auto info = curr[tid];
			auto r = rdata[tid] + dhdata[tid];
			auto mag = -4 * (double) k * info.s * r;
			for (int i = 0; i < 3; ++i)
				fdata[n * m * i + tid] = mag * info.n[i];
		};
		util::transform<128, 7>(l, n * m);
		return f;
	}

	constexpr bending(units::energy modulus, bool preferred=false) :
		modulus(modulus), preferred(preferred) {}
};

} // namespace forces
