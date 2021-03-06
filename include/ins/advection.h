#pragma once
#include <utility>
#include "util/functional.h"
#include "fd/domain.h"
#include "fd/grid.h"
#include "fd/average.h"
#include "types.h"
#include "differential.h"

namespace ins {

template <typename> class average;

template <typename ... dimension_types>
class average<fd::domain<dimension_types...>> {
public:
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
private:
	template <typename, typename R> using replace = R;

	template <typename tag_type>
	static auto
	construct(const tag_type& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto grid = [&] (const auto& dim) { return fd::grid{tag, domain, dim}; };
		const auto& dims = components(domain);
		auto grids = map(grid, dims);
		auto ave = [&] (const auto& dim, const auto& grid)
		{
			return fd::average(grid, dim);
		};
		auto row = [&] (const auto& dim) { return map(partial(ave, dim), grids); };
		return map(row, dims);
	}

	using row_type = std::tuple<replace<dimension_types, matrix>...>;
	using averages_type = std::tuple<replace<dimension_types, row_type>...>;
	averages_type averages;
public:
	template <typename ... Vs>
	auto
	operator()(Vs&& ... vs) const
	{
		using namespace util::functional;
		auto spmv = [] (const matrix& m, const vector& v) { return m * v; };
		auto&& args = std::forward_as_tuple(vs...);
		auto spmvs = [&](auto&& avs) { return map(partial(apply, spmv), zip(avs, args)); };
		return map(spmvs, averages);
	}

	template <typename tag_type>
	average(const tag_type& tag, const domain_type& domain) :
		averages(construct(tag, domain)) {}
};

// Define the central difference and average operators
//
//     Dⱼ g(x) = (g(x + heⱼ/2) - g(x - heⱼ/2)) / h,
//     Aʲ g(x) = (g(x + heⱼ/2) + g(x - heⱼ/2)) / 2,
//
// respectively. Then, the advection term ∇·(u⃗⊗u⃗) is discretized as
//
//     Hⁱ(u⃗(x)) = Dⱼ[(Aⁱuʲ) (Auʲⁱ)](x),
//
// where i and j run from 1 to d.
template <typename> class advection;

template <typename ... dimension_types>
class advection<fd::domain<dimension_types...>> {
public:
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
private:
	template <typename> using divergence_type = divergence<domain_type>;
	using average_functor_type = average<domain_type>;
	using divergence_functors_type = std::tuple<divergence_type<dimension_types>...>;

	template <typename tag_type>
	static decltype(auto)
	divergences(const tag_type& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto k = [&] (const auto m)
		{
			static constexpr auto n = decltype(m)::value;
			auto shifted = fd::shift::directionally<n>(tag);
			return divergence{shifted, domain};
		};
		return map(k, std::make_index_sequence<dimensions>{});
	}

	template <typename ... vector_types>
	auto
	tensor_product(vector_types&& ... vectors) const
	{
		using namespace util::functional;
		auto modulus = partial(apply, std::modulus<vector>());
		auto hadamard = [&] (auto pairs) { return map(modulus, std::move(pairs)); };
		auto averages = average(std::forward<vector_types>(vectors)...);
		auto complements = apply(zip, averages);
		auto pairs = map(zip, averages, std::move(complements));
		return map(hadamard, std::move(pairs));
	}

	average_functor_type average;
	divergence_functors_type divs;
public:
	template <typename ... vector_types>
	auto
	operator()(vector_types&& ... vectors) const
	{
		using namespace util::functional;
		static_assert(sizeof...(vector_types) == dimensions,
				"Number of vectors must match domain dimension");

		auto products = tensor_product(std::forward<vector_types>(vectors)...);
		return map(apply, divs, std::move(products));
	}

	template <typename tag_type>
	advection(const tag_type& tag, const domain_type& domain) :
		average(tag, domain), divs(divergences(tag, domain)) {}
};

template <typename tag_type, typename domain_type>
advection(const tag_type&, const domain_type&) -> advection<domain_type>;

} // namespace ins
