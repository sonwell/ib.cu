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

inline constexpr struct {} reference; // tag for getting reference geometry data
inline constexpr struct {} current;   // tag for getting current geometry data

// Container for geometric data:
//   * positions
//   * tangents
//   * normal
//   * second derivatives
//   * surface patch areas
template <int dims>
struct geometry {
public:
	static constexpr auto dimensions = dims;
private:
	using slice = impl::slice;
	using operators_type = operators<dimensions>;

	template <std::size_t n>
	static decltype(auto)
	compute_matrices(const std::array<slice, n>& operators, const matrix& x)
	{
		using namespace util::functional;
		auto c = [] (auto ... args) { return std::array{std::move(args)...}; };
		auto f = [&] (const slice& m) { return m * x; };
		return apply(c, map(f, operators));
	}

	static matrix
	compute_position(const operators_type& ops, const matrix& x)
	{
		const matrix& op = ops.evaluator;
		if (!op.rows() && !op.cols()) return x;
		return op * x;
	}

	static decltype(auto)
	compute_tangents(const operators_type& ops, const matrix& x)
	{
		return compute_matrices(ops.first_derivatives, x);
	}

	static decltype(auto)
	compute_second_derivatives(const operators_type& ops, const matrix& x)
	{
		return compute_matrices(ops.second_derivatives, x);
	}

	using tangents_type = decltype(compute_tangents(
				std::declval<operators_type>(),
				std::declval<matrix>()));
	using seconds_type = decltype(compute_second_derivatives(
				std::declval<operators_type>(),
				std::declval<matrix>()));
	typedef struct {
		matrix position;
		tangents_type tangents;
		matrix normal;
		seconds_type seconds;
		matrix sigma;
	} return_type;

	static return_type
	compute(const operators_type& ops, const matrix& x)
	{
		using namespace util::functional;
		auto& w = ops.weights;
		auto y = compute_position(ops, x);
		auto tangents = compute_tangents(ops, x);
		auto seconds = compute_second_derivatives(ops, x);

		auto ns = y.rows();
		auto nc = x.cols() / (dimensions + 1);
		auto count = ns * nc;
		matrix sigma{ns, nc};
		matrix n{linalg::size(y)};

		auto* wdata = w.values();
		auto tdata = map([] (const matrix& m) { return m.values(); }, tangents);

		auto* ndata = n.values();
		auto* sdata = sigma.values();

		auto f = [=] __device__ (int tid)
		{
			using tangent_type = std::array<double, dimensions+1>;
			std::array<tangent_type, dimensions> t;

			auto k = [&] (int i, tangent_type& t, const double* data)
			{
				t[i] = data[count * i + tid];
			};

			for (int i = 0; i < dimensions+1; ++i)
				map(partial(k, i), t, tdata);

			auto cross = [] (auto&& ... args)
			{
				return algo::cross(std::forward<decltype(args)>(args)...);
			};
			auto n = apply(cross, t);
			auto detf = sqrt(algo::dot(n, n));

			for (int i = 0; i < dimensions+1; ++i)
				ndata[count * i + tid] = n[i] / detf;
			sdata[tid] = wdata[tid % ns] * detf;
		};
		util::transform(f, count);

		return {std::move(y), std::move(tangents), std::move(n),
			std::move(seconds), std::move(sigma)};
	}

	void
	swap(geometry& g)
	{
		if (&g == this) return;
		std::swap(position, g.position);
		std::swap(tangents, g.tangents);
		std::swap(normal, g.normal);
		std::swap(second_derivatives, g.second_derivatives);
		std::swap(sigma, g.sigma);
	}

	geometry(return_type results) :
		position(std::move(results.position)),
		tangents(std::move(results.tangents)),
		normal(std::move(results.normal)),
		second_derivatives(std::move(results.seconds)),
		sigma(std::move(results.sigma)) {}

public:
	matrix position;
	tangents_type tangents;
	matrix normal;
	seconds_type second_derivatives;
	matrix sigma; // surface patch area

	geometry& operator=(geometry&& g) { swap(g); return *this; }

	geometry() {}
	geometry(geometry&& g) : geometry() { swap(g); }
	geometry(const operators_type& ops, const matrix& x) :
		geometry(compute(ops, x)) {}
};

}
