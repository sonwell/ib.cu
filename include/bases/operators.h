#pragma once
#include "cuda/event.h"
#include "util/sequences.h"
#include "util/functional.h"
#include "util/debug.h"
#include "algo/lu.h"
#include "types.h"
#include "differentiation.h"
#include "fill.h"

namespace bases {

template <int dims>
struct operators {
	static constexpr auto dimensions = dims;
	using sequence = util::make_sequence<int, dims>;
	using params = double[dimensions];

	template <typename ... arg_types>
	static auto
	cat(arg_types&& ... args)
	{
		static constexpr auto id = [] (auto x) { return x; };
		using namespace util::functional;
		return map(id, std::tuple_cat(std::forward<arg_types>(args)...));
	}

	template <typename rbf, typename poly>
	static matrix
	compute_evaluator(const matrix& xd, const matrix& xs, rbf phi, poly p,
			const algo::lu_factorization& lu)
	{
		return solve(lu, fill<dimensions>(xs, xd, phi, p));
	}

	template <typename rbf, typename poly, int ... ns>
	static matrix
	compute_derivative(const matrix& xd, const matrix& xs, rbf phi,
			poly p, const algo::lu_factorization& lu, partials<ns...> d)
	{
		return solve(lu, fill<dimensions>(xs, xd, diff(phi, d), diff(p, d)));
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_first_derivatives(const matrix& xd, const matrix& xs, rbf phi,
			poly p, const algo::lu_factorization& lu, util::sequence<int, ns...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p, lu, bases::d<ns>)...};
	}

	template <typename rbf, typename poly, int n, int ... ms>
	static auto
	compute_second_derivative(const matrix& xd, const matrix& xs, rbf phi,
			poly p, const algo::lu_factorization& lu, partials<n> d,
			util::sequence<int, ms...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p, lu,
				bases::d<n> * bases::d<n + ms>)...};
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_second_derivatives(const matrix& xd, const matrix& xs, rbf phi,
			poly p, const algo::lu_factorization& lu, util::sequence<int, ns...>)
	{
		return cat(compute_second_derivative(xd, xs, phi, p, lu, bases::d<ns>,
					util::make_sequence<int, dimensions-ns>())...);
	}

	static constexpr auto nfd = dims;
	static constexpr auto nsd = dims * (dims+1) / 2;
	using fdtype = std::array<matrix, nfd>;
	using sdtype = std::array<matrix, nsd>;
	struct uniform {};
	typedef struct {
		matrix evaluator;
		fdtype first;
		sdtype second;
	} ops_type;

	template <typename rbf, typename poly>
	static ops_type
	compute_operators(const matrix& xd, const matrix& xs, rbf phi, poly p)
	{
		auto lu = algo::lu(fill<dimensions>(xd, phi, p));
		return {compute_evaluator(xd, xs, phi, p, lu),
		        compute_first_derivatives(xd, xs, phi, p, lu, sequence()),
		        compute_second_derivatives(xd, xs, phi, p, lu, sequence())};
	}

	matrix evaluator;
	fdtype first_derivatives;
	sdtype second_derivatives;
	vector weights;

	operators(ops_type ops, vector weights) :
		evaluator(std::move(ops.evaluator)),
		first_derivatives(std::move(ops.first)),
		second_derivatives(std::move(ops.second)),
		weights(std::move(weights)) {}

	template <typename rbf, typename poly>
	operators(const matrix& xd, const matrix& xs, vector weights, rbf phi, poly p) :
		operators(compute_operators(xd, xs, phi, p), std::move(weights)) {}
};

} // namespace bases
