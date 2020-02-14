#pragma once
#include <utility>
#include "util/functional.h"
#include "fd/differential.h"
#include "fd/grid.h"
#include "types.h"

namespace ins {
namespace __1 {

template <typename> class differential;

template <typename ... dimension_types>
class differential<fd::domain<dimension_types...>> {
public:
	using domain_type = fd::domain<dimension_types...>;
	static constexpr auto dimensions = domain_type::dimensions;
private:
	template <typename tag_type>
	static decltype(auto)
	construct(const tag_type& tag, const domain_type& domain)
	{
		using namespace util::functional;
		auto differential = [&] (matrix& m, auto&& view)
		{
			fd::grid grid{tag, domain, view};
			m = fd::differential(grid, view);
		};
		std::array<matrix, dimensions> operators{};
		const auto& components = fd::components(domain);
		map(differential, operators, components);
		return operators;
	}

	using differentials_type = std::array<matrix, dimensions>;
protected:
	differentials_type differentials;

	template <typename tag_type>
	differential(const tag_type& tag, const domain_type& domain) :
		differentials{construct(tag, domain)} {}
};

} // namespace __1

template <typename> class divergence;

template <typename ... dimension_types>
class divergence<fd::domain<dimension_types...>> :
	public __1::differential<fd::domain<dimension_types...>> {
private:
	using domain_type = fd::domain<dimension_types...>;
	using base = __1::differential<domain_type>;
public:
	using base::dimensions;

	template <typename ... Vs>
	auto
	operator()(Vs&& ... vs) const
	{
		static_assert(sizeof...(Vs) == dimensions);
		using namespace util::functional;
		if constexpr (dimensions == 0) { return vector{1, linalg::zero}; }
		else {
			auto spmv = [] (const matrix& m, vector v) { return m * v; };
			auto vectors = std::forward_as_tuple(std::forward<Vs>(vs)...);
			auto pairs = zip(base::differentials, std::move(vectors));
			auto op = [&] (vector v, auto&& r)
			{
				return std::move(v) + apply(spmv, std::forward<decltype(r)>(r));
			};
			auto peel = [&] (auto&& f, auto&& ... r)
			{
				return foldl(op, apply(spmv, f), std::forward<decltype(r)>(r)...);
			};
			return apply(peel, std::move(pairs));
		}
	}

	template <typename tag_type, typename domain_type>
	divergence(const tag_type& tag, const domain_type& domain) :
		base(tag, domain) {}
};

template <typename tag_type, typename domain_type>
divergence(const tag_type&, const domain_type&) -> divergence<domain_type>;

template <typename> class gradient;

template <typename ... dimension_types>
class gradient<fd::domain<dimension_types...>> :
	public __1::differential<fd::domain<dimension_types...>> {
private:
	using domain_type = fd::domain<dimension_types...>;
	using base = __1::differential<domain_type>;
public:
	using base::dimensions;

	auto
	operator()(const vector& v) const
	{
		using namespace util::functional;
		auto spmv = [] (const matrix& m, const vector& v) { return m * v; };
		auto multiply = [&] (const matrix& m) { return spmv(m, v); };
		return map(multiply, base::differentials);
	}

	template <typename tag_type, typename domain_type>
	gradient(const tag_type& tag, const domain_type& domain) :
		base(tag, domain) {}
};

template <typename tag_type, typename domain_type>
gradient(const tag_type&, const domain_type&) -> gradient<domain_type>;

} // namespace ins
