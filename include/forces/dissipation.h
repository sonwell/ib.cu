#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

template <int dims>
struct velocity {
	std::array<vector<dims+1>, dims> ut;
	std::array<vector<dims+1>, nchoosek(dims+1, 2)> utt;
};

template <int dims>
struct loader<velocity<dims>> {
	using operators = bases::operators<dims>;
	struct operators_applied { std::array<matrix, dims> t; std::array<matrix, nchoosek(dims+1, 2)> tt; };

	operators_applied x;
	std::array<loader<vector<dims+1>>, dims> t;
	std::array<loader<vector<dims+1>>, nchoosek(dims+1, 2)> tt;

	constexpr velocity<dims>
	operator[](int i) const
	{
		using namespace util::functional;
		auto cons = [] (auto ... v) { return std::array<vector<dims+1>, sizeof...(v)>{std::move(v)...}; };
		auto load = [&] (const auto& l) { return l[i]; };
		return {apply(cons, map(load, t)), apply(cons, map(load, tt))};
	}

	template <std::size_t n>
	static auto
	expand(const std::array<matrix, n>& m)
	{
		using namespace util::functional;
		using load = loader<vector<dims+1>>;
		auto op = [] (const auto& ... v) { return std::array{load{v}...}; };
		return apply(op, m);
	}

	static operators_applied
	apply_ops(const operators& ops, const matrix& x)
	{
		using namespace util::functional;
		auto k = [&] (const auto& ... op) { return std::array{op * x...}; };
		return {apply(k, ops.first_derivatives), apply(k, ops.second_derivatives)};
	}

	loader(operators_applied y) :
		x{std::move(y)}, t{expand(x.t)}, tt{expand(x.tt)} {}
	loader(const operators& ops, const matrix& u) :
		loader(apply_ops(ops, u)) {}
};

template <int dims>
struct dot_metric {
	using tangents = tangents<dims>;
	using velocity = velocity<dims>;
	using subscripts = subscripts<dot_metric, 2, 1>;
	static constexpr auto n = nchoosek(dims+1, 2);
	std::array<double, n> dg;

	constexpr auto
	compute(const tangents& t, const velocity& v)
	{
		std::array<double, n> dv;
		auto ev = [&] (int i, int j) { return algo::dot(t.t[i], v.ut[j]) + algo::dot(v.ut[i], t.t[j]); };
		for (int i = 0; i < dims; ++i)
			for (int j = i; j < dims; ++j)
				dv[i * (2 * dims - i - 1) / 2 + j] = ev(i, j);
		return dv;
	}

	constexpr subscripts operator[](int i) const { return {*this, {i}}; };
	constexpr dot_metric(const tangents& t, const velocity& v) : dg{compute(t, v)} {}
	template <typename T> constexpr dot_metric(const T& t) :
		dot_metric((const tangents&) t, (const velocity&) t) {}
};

template <int dims>
constexpr decltype(auto)
resolve(const dot_metric<dims>& m, const std::array<int, 2>& subs)
{
	auto [i, j] = subs;
	if (i > j) std::swap(i, j);
	return m.dg[i * (2 * dims - i - 1) / 2 + j];
}

template <int dims>
struct dot_christoffel {
	using tangents = tangents<dims>;
	using seconds = seconds<dims>;
	using velocity = velocity<dims>;
	using subscripts = subscripts<dot_christoffel, 3, 1>;
	static constexpr auto n = nchoosek(dims+1, 2);
	std::array<double, dims * n> dg;

	constexpr auto
	compute(const tangents& t, const seconds& s, const velocity& v)
	{
		auto ev = [&] (int i, int j) { return algo::dot(v.utt[i], t.t[j]) + algo::dot(s.tt[i], v.ut[j]); };
		std::array<double, dims * n> dv;
		for (int k = 0; k < dims; ++k)
			for (int i = 0; i < n; ++i)
				dv[k * n + i] = ev(i, k);
		return dv;
	};

	constexpr subscripts operator[](int n) const { return {*this, {n}}; };
	constexpr dot_christoffel(const tangents& t, const seconds& s, const velocity& v) : dg{compute(t, s, v)} {}
	template <typename T> constexpr dot_christoffel(const T& t, const velocity& v) :
		dot_christoffel{(const tangents&) t, (const seconds&) t, v} {}
	template <typename T> constexpr dot_christoffel(const T& t) :
		dot_christoffel{(const tangents&) t, (const seconds&) t, (const velocity&) t} {}
};

template <int dims>
constexpr decltype(auto)
resolve(const dot_christoffel<dims>& c, const std::array<int, 3>& subs)
{
	auto [i, j, k] = subs;
	if (i > j) std::swap(i, j);
	return c.dg[k * nchoosek(dims+1, 2) + i * (2* dims - i - 1) / 2 + j];
}

struct dissipation {
private:
	using tangents = tangents<2>;
	using seconds = seconds<2>;
	using velocity = velocity<2>;
	using metric = metric<2>;
	using christoffel = christoffel<2>;
	using dot_metric = dot_metric<2>;
	using dot_christoffel = dot_christoffel<2>;
	using geometry = bases::geometry<2>;
	using operators = bases::operators<2>;

	struct load : loader<tangents>, loader<seconds>, loader<velocity>, loader<measure> {
		struct payload : tangents, seconds, velocity, measure {
			using tangents::t;
			using seconds::tt;
			using velocity::ut;
			using velocity::utt;
			using measure::s;
		};

		constexpr payload
		operator[](int i) const
		{
			return {{loader<tangents>::operator[](i)},
			        {loader<seconds>::operator[](i)},
			        {loader<velocity>::operator[](i)},
			        {loader<measure>::operator[](i)}};
		}

		load(const geometry& g, const operators& ops, const matrix& u) :
			loader<tangents>(g),
			loader<seconds>(g),
			loader<velocity>(ops, u),
			loader<measure>(g) {}
	};

	using payload = load::payload;

	struct helper {
		double e, f, g;
		double eu, fu, gu;
		double ev, fv, gv;
		double i, iu, iv;

		constexpr helper(const metric& m, const christoffel& c) :
			e(m[0][0]),         f(m[0][1]),                  g(m[1][1]),
			// g_{ij,k} = Γ_{ik,j} + Γ{kj,i}, Γ_{ij,k} = Γ_{ji,k}
			eu(2 * c[0][0][0]), fu(c[0][0][1] + c[0][1][0]), gu(2 * c[0][1][1]),
			ev(2 * c[0][1][0]), fv(c[0][1][1] + c[1][1][0]), gv(2 * c[1][1][1]),
			i(e * g - f * f),
			iu(eu * g + e * gu - 2 * f * fu),
			iv(ev * g + e * gv - 2 * f * fv) {}

		constexpr helper(const payload& pt) :
			helper((const metric&) pt, (const christoffel&) pt) {}
	};

	struct dot_helper {
		double de, df, dg;
		double deu, dfu, dgu;
		double dev, dfv, dgv;

		constexpr dot_helper(const dot_metric& m, const dot_christoffel& c) :
			de(m[0][0]), df(m[0][1]), dg(m[1][1]),
			// ġ_{ij,k} = Γ̇_{ik,j} + Γ̇_{kj,i}, Γ̇_{ij,k} = Γ̇_{ji,k}
			deu(2 * c[0][0][0]), dfu(c[0][0][1] + c[0][1][0]), dgu(2 * c[0][1][1]),
			dev(2 * c[0][1][0]), dfv(c[0][1][1] + c[1][1][0]), dgv(2 * c[1][1][1]) {}

		constexpr dot_helper(const payload& pt) :
			dot_helper((const dot_metric&) pt, (const dot_christoffel&) pt) {}
	};
public:
	units::unit<0, 1, -1> modulus;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object, const matrix& u) const
	{
		using bases::current;
		const auto& curr = object.geometry(current).sample;
		load deformed{curr, object.operators().sample, u};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=, kappa=(double) modulus] __device__ (int tid)
		{
			auto curr = deformed[tid];
			auto [e, f, g, eu, fu, gu, ev, fv, gv, i, iu, iv] = helper{curr};
			auto [de, df, dg, deu, dfu, dgu, dev, dfv, dgv] = dot_helper{curr};

			double pig[] = {(g * de - f * df) / i, (g * df - f * dg) / i,
			                (e * df - f * de) / i, (e * dg - f * df) / i};
			double pigu[] = {
				(gu * de + g * deu - fu * df - f * dfu - iu * pig[0]) / i,
				(gu * df + g * dfu - fu * dg - f * dgu - iu * pig[1]) / i,
				(eu * df + e * dfu - fu * de - f * deu - iu * pig[2]) / i,
				(eu * dg + e * dgu - fu * df - f * dfu - iu * pig[3]) / i
			};

			double pigv[] = {
				(gv * de + g * dev - fv * df - f * dfv - iv * pig[0]) / i,
				(gv * df + g * dfv - fv * dg - f * dgv - iv * pig[1]) / i,
				(ev * df + e * dfv - fv * de - f * dev - iv * pig[2]) / i,
				(ev * dg + e * dgv - fv * df - f * dfv - iv * pig[3]) / i
			};

			double pif[] = {
				pig[0] * g - pig[1] * f,
				0.5 * (pig[1] * e + pig[2] * g - (pig[0] + pig[3]) * f),
				pig[3] * e - pig[2] * f
			};
			double pifu[] = {
				pigu[0] * g + pig[0] * gu - pigu[1] * f - pig[1] * fu,
				0.5 * (pigu[1] * e + pig[1] * eu + pigu[2] * g + pig[2] * gu
				   - ((pigu[0] + pigu[3]) * f + (pig[0] + pig[3]) * fu)),
				pigu[3] * e + pig[3] * eu - pigu[2] * f - pig[2] * fu
			};
			double pifv[] = {
				pigv[0] * g + pig[0] * gv - pigv[1] * f - pig[1] * fv,
				0.5 * (pigv[1] * e + pig[1] * ev + pigv[2] * g + pig[2] * gv
				   - ((pigv[0] + pigv[3]) * f + (pig[0] + pig[3]) * fv)),
				pigv[3] * e + pig[3] * ev - pigv[2] * f - pig[2] * fv
			};

			double c0 = pifu[0] + pifv[1] - (iu * pif[0] + iv * pif[1]) / (2 * i);
			double c1 = pifu[1] + pifv[2] - (iu * pif[1] + iv * pif[2]) / (2 * i);

			auto& [u, v] = curr.t;
			auto& [uu, uv, vv] = curr.tt;
			auto fd = c0 * u + c1 * v + pif[0] * uu + 2 * pif[1] * uv + pif[2] * vv;

			for (int j = 0; j < 3; ++j)
				fdata[n * j + tid] = /*curr.s */ kappa / i * fd[j];
		};
		util::transform(k, n);
		return f;
	}

	constexpr dissipation(units::unit<0, 1, -1> modulus) :
		modulus(modulus) {}
};

} // namespace forces
