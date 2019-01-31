#pragma once
#include <type_traits>
#include <utility>

#include "lwps/types.h"
#include "lwps/matrix.h"
#include "util/launch.h"
#include "util/functional.h"

#include "operators.h"
#include "domain.h"
#include "identity.h"

namespace fd {
	namespace boundary_impl {
		using identity_impl::order;

		template <typename Collocation, typename View, typename Tag>
		lwps::matrix
		boundary(const View& view, const Tag& tag)
		{
			using lower_boundary_type = typename View::lower_boundary_type;
			using upper_boundary_type = typename View::upper_boundary_type;
			using corrector_type = boundary::corrector<Collocation,
				  lower_boundary_type, upper_boundary_type>;
			static constexpr auto solid_boundary = corrector_type::solid_boundary;
			corrector_type corrector(view);
			const auto rows = corrector.gridpts();
			const auto weight = corrector.boundary_weight(tag);
			const auto start = Tag::value;

			lwps::matrix result{rows, solid_boundary, solid_boundary};
			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();

			auto k = [=] __device__ (int tid)
			{
				starts[tid] = tid ? start : 0;

				if (!tid) {
					starts[rows] = 1;
					if constexpr(solid_boundary) {
						indices[0] = lwps::indexing_base;
						values[0] = weight;
					}
				}
			};
			util::transform<128, 7>(k, rows);
			return std::move(result);
		}

		using identity_impl::identity;

		template <typename Grid>
		class builder {
			private:
				using grid_type = Grid;
				using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
				template <std::size_t N> using collocation = std::tuple_element_t<N, grid_type>;
				template <std::size_t N> using order = boundary::correction::order<N>;

				template <std::size_t ... Ns, typename Views, typename View,
						 typename Tag, std::size_t N>
				static auto
				splat(const std::index_sequence<Ns...>&, const Views& views,
						const View& view, const Tag& tag, const order<N>& correction)
				{
					return std::make_tuple((std::get<Ns>(views) == view ?
								boundary<collocation<Ns>>(std::get<Ns>(views), tag) :
								identity<collocation<Ns>>(std::get<Ns>(views), correction))...);
				}
			public:
				template <typename Views, typename View, typename Tag, std::size_t N>
				static lwps::matrix
				build(const Views& views, const View& view, const Tag& tag, const order<N>& correction)
				{
					using namespace util::functional;

					auto&& components = splat(sequence(), views, view, tag, correction);
					auto&& multikron = partial(foldl, lwps::kron);
					auto&& reversed = reverse(std::move(components));
					return apply(multikron, std::move(reversed));
				}
		};
	}

	namespace boundary {
		template <typename Domain, typename View, typename Dir, std::size_t N = 0>
		lwps::matrix
		lower(const Domain& domain, const View& view, const Dir& dir,
		      const correction::order<N>& corr = correction::zeroth_order)
		{
			using operators::caller;
			using boundary_impl::builder;
			using tag_type = typename Domain::tag_type;
			using caller_type = caller<builder, tag_type, 0, Domain::ndim>;
			auto&& views = fd::dimensions(domain);
			return caller_type::call(view, views, dir, boundary_impl::lower, corr);
		}

		template <typename Domain, typename View, typename Dir, std::size_t N = 0>
		lwps::matrix
		upper(const Domain& domain, const View& view, const Dir& dir,
		      const correction::order<N>& corr = correction::zeroth_order)
		{
			using operators::caller;
			using boundary_impl::builder;
			using tag_type = typename Domain::tag_type;
			using caller_type = caller<builder, tag_type, 0, Domain::ndim>;
			auto&& views = fd::dimensions(domain);
			return caller_type::call(view, views, dir, boundary_impl::upper, corr);
		}
	}
}
