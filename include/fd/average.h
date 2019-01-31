#pragma once
#include <type_traits>
#include <iostream>
#include <utility>

#include "lwps/types.h"
#include "lwps/matrix.h"
#include "lwps/operators.h"
#include "util/launch.h"
#include "util/functional.h"

#include "operators.h"
#include "domain.h"
#include "identity.h"

namespace fd {
	namespace average_impl {
		template <typename Collocation, typename View>
		lwps::matrix
		average(View& view)
		{
			using lower_boundary_type = typename View::lower_boundary_type;
			using upper_boundary_type = typename View::upper_boundary_type;
			using corrector_type = boundary::corrector<Collocation,
				  lower_boundary_type, upper_boundary_type>;
			static constexpr auto solid_boundary = corrector_type::solid_boundary;
			static constexpr auto on_boundary = corrector_type::on_boundary;
			static constexpr auto backward = solid_boundary ^ !on_boundary;
			corrector_type corrector(view);


			const auto gridpts = view.gridpts();
			const auto cols = gridpts - solid_boundary * on_boundary;
			const auto rows = gridpts - solid_boundary * !on_boundary;

			auto nonzero = rows + cols - 1;
			lwps::matrix result{rows, cols, nonzero};

			// Periodic boundary
			// <-x-|-x-|-x-|-x-> --> x-o-x-o-x-o-x-o-> : backward, nxn, 2n-1 nonzero
			// x-o-x-o-x-o-x-o-> --> <-x-|-x-|-x-|-x-> : forward, nxn, 2n-1 nonzero
			//
			// Solid boundary
			// >-x-|-x-|-x-|-x-< --> >-o-x-o-x-o-x-o-< : forward, (n-1)xn, 2n-2 nonzero
			// >-o-x-o-x-o-x-o-< --> >-x-o-x-o-x-o-x-< : backward, nx(n-1), 2n-2 nonzero

			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();

			auto k = [=] __device__ (int tid)
			{
				if (tid < rows) starts[tid] = tid ? 2 * tid - backward : 0;
				auto loc = (tid + backward) % 2;
				auto row = (tid + backward) / 2;
				auto col = loc + row - backward;
				indices[tid] = col + lwps::indexing_base;
				values[tid] = 0.5;
				if (!tid) starts[rows] = nonzero;
			};
			util::transform<128, 7>(k, nonzero);

			if (backward)
				result += corrector.lower_stencil(rows, cols, 0.5);
			if (on_boundary)
				result += corrector.upper_stencil(rows, cols, 0.5);
			return std::move(result);
		}

		using identity_impl::identity;

		template <typename Grid>
		class builder {
			private:
				using grid_type = Grid;
				using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
				template <std::size_t N> using collocation = std::tuple_element_t<N, grid_type>;
				static constexpr auto& order = boundary::correction::zeroth_order;

				template <std::size_t ... Ns, typename Views, typename View>
				static auto
				splat(const std::index_sequence<Ns...>&, const Views& views, const View& view)
				{
					return std::make_tuple((std::get<Ns>(views) == view ?
								average<collocation<Ns>>(std::get<Ns>(views)) :
								identity<collocation<Ns>>(std::get<Ns>(views), order))...);
				}
			public:
				template <typename Views, typename View>
				static lwps::matrix
				build(const Views& views, const View& view)
				{
					using namespace util::functional;

					auto&& components = splat(sequence(), views, view);
					auto&& multikron = partial(foldl, lwps::kron);
					auto&& reversed = reverse(std::move(components));
					return apply(multikron, std::move(reversed));
				}
		};
	}

	static constexpr struct __average_functor {
		template <typename Domain, typename View, typename Dir>
		lwps::matrix
		operator()(const Domain& domain, const View& view, const Dir& dir) const
		{
			using operators::caller;
			using average_impl::builder;
			using tag_type = typename Domain::tag_type;
			using caller_type = caller<builder, tag_type, 0, Domain::ndim>;
			auto&& views = fd::dimensions(domain);
			return caller_type::call(view, views, dir);
		}

		constexpr __average_functor () {}
	} average;
}
