#pragma once
#include <array>
#include "util/math.h"
#include "differentiation.h"

namespace bases {
namespace detail {

template <int ... ns>
using keep = decltype(util::unique(util::sort(
				util::sequence<int, ns...>())));

template <int d, int m>
constexpr inline auto
subone(util::sequence<int> seq)
{
	return seq;
}

template <int d, int m, int n, int ... ns>
constexpr inline auto
subone(util::sequence<int, n, ns...> seq)
{
	if constexpr (d == m) {
		if constexpr (n <= 0) return seq;
		else return util::sequence<int, n-1, ns...>{};
	}
	else return util::concat(util::sequence<int, n>{},
		subone<d, m+1>(util::sequence<int, ns...>{}));
}

template <int ... ns>
constexpr inline auto
subeach(util::sequence<int, ns...> seq, partials<> p)
{
	return seq;
}

template <int ... ns, int d, int ... ds>
constexpr inline auto
subeach(util::sequence<int, ns...> seq, partials<d, ds...> p)
{
	return subone<d, 0>(subeach(seq, partials<ds...>{}));
}

template <int ... ns, int ... ds>
constexpr inline int
coefficient(util::sequence<int, ns...>, partials<ds...>)
{
	int exponents[] = {ns..., 0};
	int partials[] = {ds...};
	int coefficient = 1.0;
	for (int p: partials) {
		coefficient *= exponents[p];
		exponents[p]--;
	}
	return coefficient;
}

constexpr inline auto
multiindex(int row, int col, int dims)
{
	using namespace util::math;
	if (dims == 0) return 0;
	auto n = dims - 1;
	for (int i = 0;; ++i) {
		auto count = nchoosek(n + i, i);
		if (row < count) {
			auto v = multiindex(row, col-1, n);
			return col > 0 ? v : col < 0 ? i : (i - v);
		}
		row -= count;
	}
}

template <int degree, int dims>
class info {
	template <typename ...> struct container {};
	static constexpr auto n = util::math::nchoosek(degree+dims, degree);
	using row_seq = util::make_sequence<int, n>;
	using col_seq = util::make_sequence<int, dims>;

	template <int row, int ... cols>
	static constexpr auto
	make_row(util::sequence<int, row>, util::sequence<int, cols...>)
	{
		return util::sequence<int, multiindex(row, cols, dims)...>{};
	}

	template <int ... rows>
	static constexpr auto
	make_col(util::sequence<int, rows...>)
	{
		return container<decltype(make_row(util::sequence<int, rows>{}, col_seq{}))...>{};
	}

	template <typename ... sequences, int ... ds>
	static constexpr auto
	coefficients(container<sequences...>, partials<ds...>)
	{
		return util::sequence<int, coefficient(sequences{},
				partials<ds...>{})...>{};
	}

	template <typename ... sequences, int ... ds>
	static constexpr auto
	exponents(container<sequences...>, partials<ds...>)
	{
		return util::concat(subeach(sequences{},
					partials<ds...>{})...);
	}

	using type = decltype(make_col(row_seq()));
public:
	template <int ... ds>
	static constexpr auto
	coefficients(partials<ds...> p)
	{
		return coefficients(type{}, p);
	}

	template <int ... ds>
	static constexpr auto
	exponents(partials<ds...> p)
	{
		return exponents(type{}, p);
	}
};

} // namespace detail

// polynomials<d> accepts an array of n parameters and evaluates each of the
// monomials x_1^a x_2^b ... with a + b + ... <= d.
template <int degree>
class polynomials : differentiable {
private:
	template <int n, int ... exps, int ... coeffs>
	constexpr auto
	eval(const double (&xs)[n], util::sequence<int, exps...>,
			util::sequence<int, coeffs...>) const
	{
		using util::math::pow;
		constexpr int exponents[] = {exps...};
		constexpr int coefficients[] = {coeffs...};
		constexpr int np = sizeof...(coeffs);

		std::array<double, np> values;
		for (int i = 0; i < np; ++i) {
			double v = coefficients[i];
			for (int j = 0; j < n; ++j)
				v *= pow(xs[j], exponents[j + n * i]);
			values[i] = v;
		}
		return values;
	}
public:
	template <int n, int ... ds>
	constexpr auto
	operator()(const double (&xs)[n], partials<ds...> p = {}) const
	{
		using info_type = detail::info<degree, n>;
		return eval(xs, info_type::exponents(p), info_type::coefficients(p));
	}
};

template <int, typename> class polynomial_subset;


// polynomial_subset<d, util::sequence<i, j, ...>> accepts an array of n
// parameters, but uses them to evaluate just the monomials x_i^a x_j^b ...,
// with a + b + ... <= d.
template <int degree, int ... ns>
class polynomial_subset<degree, util::sequence<int, ns...>> : public polynomials<degree> {
private:
	using keep_type = detail::keep<ns...>;

	static constexpr auto
	index(int d, util::sequence<int>)
	{
		return 0;
	}

	template <int k, int ... ks>
	static constexpr auto
	index(int d, util::sequence<int, k, ks...>)
	{
		return d == k ? 1 + sizeof...(ks) :
			index(d, util::sequence<int, ks...>{});
	}

	static constexpr auto
	index(int d)
	{
		return keep_type::size() - index(d, keep_type{});
	}

	template <int ... ds>
	static constexpr auto
	remap(partials<ds...>)
	{
		return partials<index(ds)...>{};
	}

	template <int n, int ... ds, int ... ks>
	constexpr auto
	eval(const double (&xs)[n], partials<ds...> p, util::sequence<int, ks...>) const
	{
		constexpr decltype(remap(p)) q;
		const double xp[] = {xs[ks]...};
		return polynomials<degree>::operator()(xp, q);
	}
public:
	template <int n, int ... ds>
	constexpr auto
	operator()(const double (&xs)[n], partials<ds...> p = partials<>()) const
	{
		return eval(xs, p, keep_type{});
	}
};

struct no_polynomials : differentiable {
	template <int n, int ... ds>
	constexpr auto
	operator()(const double (&)[n], partials<ds...> p = partials<>{}) const
	{
		return std::array<double, 0>{};
	}
};

template <typename> struct is_polynomial_basis : std::false_type {};

template <int degree>
struct is_polynomial_basis<polynomials<degree>> : std::true_type {};

template <>
struct is_polynomial_basis<no_polynomials> : std::true_type {};

template <int degree, int ... ns>
struct is_polynomial_basis<polynomial_subset<degree, util::sequence<int, ns...>>> : std::true_type {};

template <typename T>
inline constexpr bool is_polynomial_basis_v = is_polynomial_basis<T>::value;

namespace meta {

template <typename T> concept polynomial = is_polynomial_basis_v<T>;

}

} // namespace bases
