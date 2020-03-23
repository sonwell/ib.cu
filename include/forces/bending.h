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

struct bending {
	units::energy modulus;

	decltype(auto)
	multiply(cublas::handle& k, const matrix& op, const matrix& x) const
	{
		static constexpr double alpha = 1.0;
		static constexpr double beta = 0.0;
		matrix r{op.cols(), x.cols()};
		cublas::operation op_a = cublas::operation::transpose;
		cublas::operation op_b = cublas::operation::non_transpose;
		cublas::gemm(k, op_a, op_b, op.cols(), x.cols(), x.rows(),
				&alpha, op.values(), op.rows(), x.values(), x.rows(),
				&beta, r.values(), r.rows());
		return r;
	}

	decltype(auto)
	mean_curvature(const loader<2>& current) const
	{
		auto [n, m] = current.size();
		matrix h{n, m};
		auto* hdata = h.values();
		auto k = [=] __device__ (int tid)
		{
			auto curr = current(tid);
			auto [e, f, g, detg] = first_fundamental_form{curr};
			auto [l, m, n, detb] = second_fundamental_form{curr};
			auto h = (l * g + n * e - 2 * m * f) / (2 * detg);
			hdata[tid] = h;
		};
		util::transform(k, n * m);
		return h;
	}

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using bases::current;
		cublas::handle handle;
		auto [d2d, d2s] = object.operators();
		auto [currd, currs] = object.geometry(current);
		loader loadd{currd};
		loader loads{currs};

		auto lh = mean_curvature(loadd);
		auto hu = multiply(handle, d2d.first_derivatives[0], lh);
		auto hv = multiply(handle, d2d.first_derivatives[1], lh);

		auto sd = loadd.size();
		auto nd = sd.rows * sd.cols;
		auto* udata = hu.values();
		auto* vdata = hv.values();
		auto k = [=] __device__ (int tid)
		{
			auto curr = loadd(tid);
			auto [e, f, g, detg] = first_fundamental_form{curr};
			auto detf = sqrt(detg);
			auto u = udata[tid];
			auto v = vdata[tid];

			auto a = (+g * u - f * v) / detf;
			auto b = (-f * u + e * v) / detf;

			udata[tid] = a;
			vdata[tid] = b;
		};
		util::transform(k, nd);

		lh = multiply(handle, d2s.first_derivatives[0], hu)
		   + multiply(handle, d2s.first_derivatives[1], hv);
		matrix f{linalg::size(currs.position)};

		auto ss = loads.size();
		auto ns = ss.rows * ss.cols;
		auto* ldata = lh.values();
		auto* fdata = f.values();
		auto l = [=, modulus=modulus] __device__ (int tid)
		{
			auto curr = loads(tid);
			auto [e, f, g, detg] = first_fundamental_form{curr};
			auto [l, m, n, detb] = second_fundamental_form{curr};
			auto detf = sqrt(detg);
			auto lh = ldata[tid];
			auto h = (l * g + n * e - 2 * m * f) / (2 * detg);
			auto k = detb / detg;

			auto mag = -curr.s * modulus * (lh / detf + 2 * h * (h * h - k));
			for (int i = 0; i < 3; ++i)
				fdata[ns * i + tid] = mag * curr.n[i];
		};
		util::transform(l, ns);
		return f;
	}

	constexpr bending(units::energy modulus) :
		modulus(modulus) {}
};

} // namespace forces
