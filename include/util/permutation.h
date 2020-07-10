#pragma once
#include "sequences.h"
#include "container.h"

// Computes all permutations of 0 ... n-1 at compile time.
//
//     using p = permutations<3>;
//     // p = container<permutation< 1, 0, 1, 2>,
//     //               permutation<-1, 0, 2, 1>,
//     //               permutation< 1, 1, 2, 0>,
//     //               permutation<-1, 1, 0, 2>,
//     //               permutation< 1, 2, 0, 1>,
//     //               permutation<-1, 2, 1, 0>>
//     //                   sign ~~~~^  ^~~^~~^~~~~ permutation
//

namespace util {

template <int sigma, int ... order>
struct permutation {
	static constexpr auto sign = sigma;
	using type = sequence<int, order...>;
};

template <int n>
struct make_permutation {
	using base = make_permutation<n-1>;

	template <int m, int s, int ... ms>
	static constexpr auto
	expand(permutation<s, ms...>)
	{
		constexpr auto count = sizeof...(ms);
		constexpr auto change = count % 2 ? m % 2 ? -1 : 1 : 1;
		return permutation<s * change, m, (m + 1 + ms) % n...>{};
	}

	template <int ... ns, int s, int ... ms>
	static constexpr auto
	expand(sequence<int, ns...>, permutation<s, ms...> p)
	{
		return container<decltype(expand<ns>(p))...>{};
	}

	template <typename ... permutation>
	static constexpr auto
	expand(container<permutation...>)
	{
		using sequence = make_sequence<int, n>;
		return concat(expand(sequence(), permutation())...);
	}

	template <typename ... permutation>
	static constexpr auto
	signs(container<permutation...>)
	{
		return sequence<int, permutation::sign...>{};
	}

	template <typename ... permutation>
	static constexpr auto
	indices(container<permutation...>)
	{
		return concat(util::sequence<int>{},
				typename permutation::type{}...);
	}

	using type = decltype(expand(typename base::type()));
	using sign_type = decltype(signs(type{}));
	using index_type = decltype(indices(type{}));
};

template <>
struct make_permutation<0> {
	using type = container<permutation<1>>;
};

template <int n>
using permutations = typename make_permutation<n>::type;

}
