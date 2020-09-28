#pragma once
#include <iostream>
#include "cuda/event.h"
#include "util/sequences.h"
#include "util/functional.h"
#include "util/debug.h"
#include "algo/lu.h"
#include "algo/qr.h"
#include "types.h"
#include "differentiation.h"
#include "fill.h"
#include "rbf.h"
#include "polynomials.h"

namespace bases {

// Constructs geometric operators (eval, differentiate, etc.) used in `geometry`
// (geometry.h), except for surface patch areas, which are handled on a
// surface-by-surface basis.
template <int dims>
class operators {
public:
	static constexpr auto dimensions = dims;
private:
	using sequence = util::make_sequence<int, dims>;
	static constexpr sequence seq;

	static constexpr auto nfd = dims;
	static constexpr auto nsd = dims * (dims+1) / 2;
	using fdtype = std::array<matrix, nfd>;
	using sdtype = std::array<matrix, nsd>;

	template <typename rbf, typename poly>
	static matrix
	compute_evaluator(const matrix& xd, const matrix& xs, rbf phi, poly p)
	{
		return fill<dimensions>(xs, xd, phi, p);
	}

	template <typename rbf, typename poly, int ... ns>
	static matrix
	compute_derivative(const matrix& xd, const matrix& xs, rbf phi, poly p,
			partials<ns...> d)
	{
		return fill<dimensions>(xs, xd, diff(phi, d), diff(p, d));
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_first_derivatives(const matrix& xd, const matrix& xs, rbf phi, poly p,
			util::sequence<int, ns...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p, bases::d<ns>)...};
	}

	template <typename rbf, typename poly, int n, int ... ms>
	static auto
	compute_second_derivative(const matrix& xd, const matrix& xs, rbf phi, poly p,
			partials<n> d, util::sequence<int, ms...>)
	{
		return std::array{compute_derivative(xd, xs, phi, p,
				bases::d<n> * bases::d<n + ms>)...};
	}

	template <typename rbf, typename poly, int ... ns>
	static auto
	compute_second_derivatives(const matrix& xd, const matrix& xs, rbf phi, poly p,
			util::sequence<int, ns...>)
	{
		auto ops = std::tuple_cat(compute_second_derivative(xd, xs, phi, p, bases::d<ns>,
					util::make_sequence<int, dimensions-ns>())...);
		auto k = [] (auto ... args) { return std::array{std::move(args)...}; };
		return apply(k, std::move(ops));
	}

	typedef struct {
		matrix evaluator;
		fdtype first;
		sdtype second;
	} ops_type;

	template <typename interp, typename eval, typename poly>
	static ops_type
	compute_operators(const matrix& xd, const matrix& xs,
			interp phi, eval psi, poly p)
	{
		ops_type ops;
		ops.evaluator = compute_evaluator(xd, xs, phi, p);
		ops.first = compute_first_derivatives(xd, xs, phi, p, seq);
		ops.second = compute_second_derivatives(xd, xs, phi, p, seq);
		return ops;
	}

	operators(int nd, int ns, ops_type ops, vector weights) :
		data_sites(nd), sample_sites(ns),
		restrictor(algo::qr(std::move(ops.evaluator))),
		first_derivatives(std::move(ops.first)),
		second_derivatives(std::move(ops.second)),
		weights(std::move(weights)) {}
public:
	int data_sites;
	int sample_sites;
	algo::qr_factorization restrictor;
	fdtype first_derivatives;
	sdtype second_derivatives;
	vector weights;

	template <meta::rbf interp, meta::rbf eval, meta::polynomial poly>
	operators(const matrix& xd, const matrix& xs, vector weights, interp phi, eval psi, poly p) :
		operators(xd.rows(), xs.rows(), compute_operators(xd, xs, phi, psi, p), std::move(weights)) {}

	template <meta::rbf rbf, meta::polynomial poly>
	operators(const matrix& xd, const matrix& xs, vector weights, rbf phi, poly p) :
		operators(xd, xs, std::move(weights), phi, phi, p) {}
};

} // namespace bases
