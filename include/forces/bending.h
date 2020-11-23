#pragma once
#include "cublas/handle.h"
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct bending {
private:
	using tangents = tangents<2>;
	using seconds = seconds<2>;
	using normal = normal<2>;
	using metric = metric<2>;
	using christoffel = christoffel<2>;
	using curvature = curvature<2>;
	using geometry = bases::geometry<2>;
	using operators = bases::operators<2>;

	struct load : loader<tangents>, loader<seconds>, loader<normal>, loader<measure> {
		struct payload : tangents, seconds, normal, measure {
			using tangents::t;
			using seconds::tt;
			using normal::n;
			using measure::s;
		};

		constexpr payload
		operator[](int i) const
		{
			return {{loader<tangents>::operator[](i)},
			        {loader<seconds>::operator[](i)},
			        {loader<normal>::operator[](i)},
			        {loader<measure>::operator[](i)}};
		}

		load(const bases::geometry<2>& g) :
			loader<tangents>(g),
			loader<seconds>(g),
			loader<normal>(g),
			loader<measure>(g) {}
	};

	using payload = load::payload;

	struct mean {
		double h;

		static constexpr auto
		compute(const metric& me, const curvature& b)
		{
			auto [l, k, m] = b.b;
			auto [e, f, g] = me.g;
			return (l * g + m * e - 2 * k * f) / (2 * (e * g - f * f));
		}

		constexpr mean(const payload& pt) :
			h(compute((const metric&) pt, (const curvature&) pt)) {}
	};

	struct gaussian {
		double k;

		static constexpr auto
		compute(const metric& me, const curvature& b)
		{
			auto [l, k, m] = b.b;
			auto [e, f, g] = me.g;
			return (l * m - k * k) / (e * g - f * f);
		}

		constexpr gaussian(const payload& pt) :
			k(compute((const metric&) pt, (const curvature&) pt)) {}
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

	struct hload {
		struct operators_applied { std::array<matrix, 2> t; std::array<matrix, 3> tt; };

		operators_applied x;
		std::array<loader<double>, 2> t;
		std::array<loader<double>, 3> tt;

		struct payload { std::array<double, 2> t; std::array<double, 3> tt; };

		static operators_applied
		apply_ops(const operators& ops, const matrix& x)
		{
			using namespace util::functional;
			auto k = [&] (const auto& ... op) { return std::array{op * x...}; };
			return {apply(k, ops.first_derivatives), apply(k, ops.second_derivatives)};
		}

		constexpr payload
		operator[](int i) const
		{
			return {{t[0][i], t[1][i]}, {tt[0][i], tt[1][i], tt[2][i]}};
		}

		hload(operators_applied y) :
			x{std::move(y)}, t{x.t[0], x.t[1]}, tt{x.tt[0], x.tt[1], x.tt[2]} {}
		hload(const operators& ops, const matrix& x) : hload(apply_ops(ops, x)) {}
	};

	template <typename f_type>
	auto
	compute(const geometry& g0, const geometry& g, const f_type& f) const
	{
		auto [n, m] = linalg::size(g.position);
		auto l = n * m / 3;
		matrix h{n, m / 3};

		load orig{g0}, curr{g};
		auto *hdata = h.values();
		auto k = [=] __device__ (int tid, auto f)
		{
			hdata[tid] = f(orig[tid], curr[tid]);
		};
		util::transform<128, 7>(k, l, f);
		return h;
	}

public:
	units::energy modulus;
	bool preferred;

	template <typename object_type>
	decltype(auto)
	laplacian_mean_curvature(const object_type& object) const
	{
		using bases::current;
		using bases::reference;
		auto& g0d = object.geometry(reference).data;
		auto& gd = object.geometry(current).data;
		auto k = [p=preferred] __device__ (const payload& d0, const payload& d)
		{
			auto h = mean{d}.h;
			if (p) h -= mean{d0}.h;
			return h;
		};
		auto h = compute(g0d, gd, k);

		auto& gs = object.geometry(current).sample;
		auto [n, m] = linalg::size(gs.position);
		auto lh = matrix{n, m/3};

		hload data{object.operators().sample, h};
		load curr{gs};
		auto* ldata = lh.values();
		auto l = [=] __device__ (int tid)
		{
			auto h = data[tid];
			auto x = curr[tid];
			auto [e, f, g, eu, fu, gu, ev, fv, gv, i, iu, iv] = helper{x};
			auto [hu, hv] = h.t;
			auto [huu, huv, hvv] = h.tt;

			auto lh = (((- iu * g + iv * f) / (2 * i) + gu - fv) / i * hu +
			           ((+ iu * f - iv * e) / (2 * i) - fu + ev) / i * hv) +
			          g / i * huu - 2 * f / i * huv + e / i * hvv;
			ldata[tid] = lh;
		};
		util::transform<127, 3>(l, n*m/3);
		return lh;
	}

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		/* F_bend = -4κ(Δ(H-H₀) + 2(H-H₀)(H²-K + HH₀))n̂ */

		using bases::current;
		using bases::reference;

		auto& curr = object.geometry(current).sample;
		auto& orig = object.geometry(reference).sample;

		auto k = [=, p=preferred] __device__ (const payload& d0, const payload& d)
		{
			auto h = mean{d}.h;
			auto k = gaussian{d}.k;
			auto dh = h;
			auto sh = h * h - k;
			if (p) {
				auto h0 = mean{d0}.h;
				dh -= h0;
				sh += h * h0;
			}
			return dh * sh;
		};

		auto b = laplacian_mean_curvature(object) + 2 * compute(orig, curr, k);
		auto [n, m] = linalg::size(curr.position);
		matrix f{n, m};
		load deformed{curr};
		auto* fdata = f.values();
		auto* bdata = b.values();
		auto l = [=, ns = n*m/3, modulus=modulus] __device__ (int tid)
		{
			auto curr = deformed[tid];
			auto mag = -4 * curr.s * modulus * bdata[tid];
			for (int i = 0; i < 3; ++i)
				fdata[ns * i + tid] = mag * curr.n[i];
		};
		util::transform(l, n * m/3);
		return f;
	}

	constexpr bending(units::energy modulus, bool preferred=false) :
		modulus(modulus), preferred(preferred) {}
};

} // namespace forces
