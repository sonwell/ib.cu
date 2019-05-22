#pragma once
#include <array>
#include "cublas/handle.h"
#include "cublas/operation.h"
#include "cublas/gemm.h"
#include "algo/cross.h"
#include "algo/dot.h"
#include "types.h"
#include "operators.h"

namespace bases {

template <int> struct geometry;

template <>
struct geometry<2> {
private:
	static matrix
	apply(cublas::handle& k, const matrix& op, const matrix& x)
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

	static matrix
	compute_position(cublas::handle& k, const operators<2>& ops, const matrix& x)
	{
		return apply(k, ops.evaluator, x);
	}

	static std::array<matrix, 2>
	compute_tangents(cublas::handle& k, const operators<2>& ops, const matrix& x)
	{
		return {apply(k, ops.first_derivatives[0], x),
		        apply(k, ops.first_derivatives[1], x)};
	}

	static std::array<matrix, 3>
	compute_second_derivatives(cublas::handle& k, const operators<2>& ops, const matrix& x)
	{
		return {apply(k, ops.second_derivatives[0], x),
		        apply(k, ops.second_derivatives[1], x),
		        apply(k, ops.second_derivatives[2], x)};
	}

	using tangents_type = std::array<matrix, 2>;
	using seconds_type = std::array<matrix, 3>;
	typedef struct {
		matrix position;
		tangents_type tangents;
		matrix normal;
		seconds_type seconds;
		vector sigma;
	} return_type;

	static return_type
	compute(const operators<2>& ops, const matrix& x)
	{
		cublas::handle hdl;
		auto&& w = ops.weights;
		auto&& y = compute_position(hdl, ops, x);
		auto&& [t1, t2] = compute_tangents(hdl, ops, x);
		auto&& [t11, t12, t22] = compute_second_derivatives(hdl, ops, x);

		auto m = w.rows();
		auto count = m * x.cols() / 3;
		vector sigma{count};
		matrix n{count, 3}; // 3 = 3D

		auto* wdata = w.values();
		auto* udata = t1.values();
		auto* vdata = t2.values();

		auto* ndata = n.values();
		auto* sdata = sigma.values();

		auto f = [=] __device__ (int tid)
		{
			util::array<double, 3> u;
			util::array<double, 3> v;

			for (int i = 0; i < 3; ++i) {
				u[i] = udata[count * i + tid];
				v[i] = vdata[count * i + tid];
			}

			auto&& n = algo::cross(u, v);
			auto detf = sqrt(algo::dot(n, n));

			for (int i = 0; i < 3; ++i)
				ndata[count * i + tid] = n[i] / detf;
			sdata[tid] = wdata[tid % m] * detf;
		};
		util::transform(f, count);

		return {std::move(y), {std::move(t1), std::move(t2)}, std::move(n),
			{std::move(t11), std::move(t12), std::move(t22)}, std::move(sigma)};
	}

	geometry(return_type results) :
		position(std::move(results.position)),
		tangents(std::move(results.tangents)),
		normal(std::move(results.normal)),
		second_derivatives(std::move(results.seconds)),
		sigma(std::move(results.sigma)) {}

public:
	static constexpr auto dimensions = 2;

	matrix position;
	std::array<matrix, 2> tangents;
	matrix normal;
	std::array<matrix, 3> second_derivatives;
	vector sigma;

	geometry(const operators<2>& ops, const matrix& x) :
		geometry(compute(ops, x)) {}
};

template <int n> geometry(const operators<n>&, const matrix&) -> geometry<n>;

}
