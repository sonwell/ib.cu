#pragma once
#include "cublas/handle.h"
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

// ds^2 = e da^1 + 2f da db + g db^2
// Tells us stuff about lengths / areas
struct first_fundamental_form {
	double e, f, g, i;

	constexpr first_fundamental_form(const info<2>& load) :
		e(algo::dot(load.t[0], load.t[0])),
		f(algo::dot(load.t[0], load.t[1])),
		g(algo::dot(load.t[1], load.t[1])),
		i(e * g - f * f) {}
};

// l da^2 + 2 m da db + n db^2
// Tells us stuff about curvature
struct second_fundamental_form {
	double l, m, n, ii;

	constexpr second_fundamental_form(const info<2>& load) :
		l(algo::dot(load.tt[0], load.n)),
		m(algo::dot(load.tt[1], load.n)),
		n(algo::dot(load.tt[2], load.n)),
		ii(l * n - m * m) {}
};

// Shape function:
//     [ l m ] /[ e f ]
//     [ m n ]/ [ f g ]

constexpr double
mean_curvature(const first_fundamental_form& fff,
		const second_fundamental_form& sff)
{
	auto [e, f, g, i] = fff;
	auto [l, m, n, ii] = sff;
	return (l * g + n * e - 2 * m * f) / (2 * i);
}

constexpr double
gaussian_curvature(const first_fundamental_form& fff,
		const second_fundamental_form& sff)
{
	auto [e, f, g, i] = fff;
	auto [l, m, n, ii] = sff;
	return ii / i;
}

struct bending {
	units::energy modulus;
	bool preferred;

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

	template <typename object_type>
	decltype(auto)
	laplacian_mean_curvature(cublas::handle& handle,
			const object_type& object) const
	{
		using fff = first_fundamental_form;
		using sff = second_fundamental_form;
		using bases::current;
		using bases::reference;
		auto [d2d, d2s] = object.operators();
		auto& curr = object.geometry(current).data;
		auto& orig = object.geometry(reference).data;

		loader loadc{curr};
		loader loado{orig};

		auto [n, m] = loadc.size();
		matrix lh{n, m};
		auto* hdata = lh.values();

		auto h = [=, p=preferred] __device__ (int tid)
		{
			auto curr = loadc[tid];
			auto cfff = fff{curr};
			auto csff = sff{curr};
			auto dh = mean_curvature(cfff, csff);
			if (p) {
				auto orig = loado[tid];
				auto offf = fff{orig};
				auto osff = sff{orig};
				dh -= mean_curvature(offf, osff);
			}
			hdata[tid] = dh;
		};
		util::transform(h, n * m);

		auto hu = multiply(handle, d2d.first_derivatives[0], lh);
		auto hv = multiply(handle, d2d.first_derivatives[1], lh);

		auto* udata = hu.values();
		auto* vdata = hv.values();

		auto k = [=] __device__ (int tid)
		{
			auto curr = loadc[tid];
			auto [e, f, g, i] = first_fundamental_form{curr};
			auto detf = sqrt(i);
			auto u = udata[tid];
			auto v = vdata[tid];

			auto a = (+g * u - f * v) / detf;
			auto b = (-f * u + e * v) / detf;

			udata[tid] = a;
			vdata[tid] = b;
		};
		util::transform(k, n * m);

		return multiply(handle, d2s.first_derivatives[0], hu)
		   + multiply(handle, d2s.first_derivatives[1], hv);
	}

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		/* F_bend = -4κ(Δ(H-H₀) - 2(H-H₀)(H²-K))n̂ */

		using bases::current;
		using bases::reference;

		cublas::handle handle;
		auto& curr = object.geometry(current).sample;
		auto& orig = object.geometry(reference).sample;
		loader loadc{curr};
		loader loado{orig};

		auto [n, m] = loadc.size();
		matrix lh = laplacian_mean_curvature(handle, object);
		matrix f{linalg::size(curr.position)};
		auto* ldata = lh.values();
		auto* fdata = f.values();

		auto l = [=, ns=n*m, modulus=modulus, p=preferred] __device__ (int tid)
		{
			auto curr = loadc[tid];
			auto cfff = first_fundamental_form{curr};
			auto csff = second_fundamental_form{curr};
			auto h = mean_curvature(cfff, csff);
			auto k = gaussian_curvature(cfff, csff);

			auto detf = sqrt(cfff.i);
			auto lh = ldata[tid];
			auto dh = h;
			if (p) {
				auto orig = loado[tid];
				auto offf = first_fundamental_form{orig};
				auto osff = second_fundamental_form{orig};
				dh -= mean_curvature(offf, osff);
			}

			auto mag = -4 * curr.s * modulus * (lh / detf + 2 * dh * (h * h - k));
			for (int i = 0; i < 3; ++i)
				fdata[ns * i + tid] = mag * curr.n[i];
		};
		util::transform(l, n * m);
		return f;
	}

	constexpr bending(units::energy modulus, bool preferred=false) :
		modulus(modulus), preferred(preferred) {}
};

} // namespace forces
