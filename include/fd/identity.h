#pragma once
#include <type_traits>
#include <utility>
#include <tuple>
#include "grid.h"
#include "boundary_corrector.h"
#include "domain.h"
#include "operators.h"
#include "lwps/matrix.h"
#include "lwps/kron.h"
#include "util/launch.h"
#include "util/functional.h"

namespace fd {
	namespace identity_impl {
		using fd::boundary::correction::order;

		template <typename Collocation, typename View, std::size_t N>
		lwps::matrix
		identity(const View& view, const order<N>& correction)
		{
			using lower_boundary_type = typename View::lower_boundary_type;
			using upper_boundary_type = typename View::upper_boundary_type;
			using corrector_type = boundary::corrector<Collocation,
				  lower_boundary_type, upper_boundary_type>;
			corrector_type corrector(view);
			const auto rows = corrector.gridpts();
			lwps::matrix result{rows, rows, rows};

			lwps::index_type* starts = result.starts();
			lwps::index_type* indices = result.indices();
			lwps::value_type* values = result.values();
			auto [lw, uw] = corrector.identity_weights(correction);

			auto k = [=, lw = lw, uw = uw] __device__ (int tid)
			{
				starts[tid] = tid;
				indices[tid] = tid + lwps::indexing_base;
				values[tid] = 1.0 + (tid > 0 ? (tid < rows-1 ? 0 : uw) : lw);
				if (!tid) starts[rows] = rows;
			};
			util::transform<128, 7>(k, rows);
			return std::move(result);
		}

		template <typename Grid>
		class builder {
			private:
				using grid_type = Grid;
				using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
				template <std::size_t N> using collocation = std::tuple_element_t<N, grid_type>;

				template <std::size_t ... Ns, typename Views, std::size_t N>
				static auto
				splat(const std::index_sequence<Ns...>&, const Views& views,
						const order<N>& correction)
				{
					return std::make_tuple(identity<collocation<Ns>>(
								std::get<Ns>(views), correction)...);
				}
			public:
				template <typename Views, std::size_t N>
				static lwps::matrix
				build(const Views& views, const order<N>& correction)
				{
					using namespace util::functional;

					auto&& identities = splat(sequence(), views, correction);
					auto&& multikron = partial(foldl, lwps::kron);
					auto&& reversed = reverse(std::move(identities));
					return apply(multikron, std::move(reversed));
				}
		};
	}

	template <typename domain_type, typename view_type, std::size_t n = 0>
	auto
	identity(const domain_type& domain, const view_type& view,
			const boundary::correction::order<n>& correction = boundary::correction::zeroth_order)
	{
		using operators::caller;
		using identity_impl::builder;
		using tag_type = typename domain_type::tag_type;
		static constexpr auto dimensions = domain_type::ndim;
		using caller_type = caller<builder, tag_type, 0, dimensions>;
		auto&& views = fd::dimensions(domain);
		return caller_type::call(view, views, correction);
	}

	template <typename domain_type, std::size_t n = 0>
	auto
	identity(const domain_type& domain,
			const boundary::correction::order<n>& correction = boundary::correction::zeroth_order)
	{
		static_assert(grid::is_uniform_v<domain_type::tag_type>,
				"the 2-argument variant of fd::identity requires a uniform grid (cell- or vertex-centered)");
		return identity(domain, std::get<0>(dimensions(domain)), correction);
	}

	/*static constexpr struct __identity_functor {
		template <std::size_t N> using order = boundary::correction::order<N>;
		static constexpr auto default_order = boundary::correction::zeroth_order;

		template <typename Domain, typename View, std::size_t N = 0>
		lwps::matrix
		operator()(const Domain& domain, const View& view,
				const order<N>& correction = default_order) const
		{
			using operators::caller;
			using identity_impl::builder;
			using tag_type = typename Domain::tag_type;
			using caller_type = caller<builder, tag_type, 0, Domain::ndim>;
			auto&& views = dimensions(domain);
			return caller_type::call(view, views, correction);
		}

		template <typename Domain, std::size_t N = 0>
		std::enable_if_t<Domain::tag_type::is_uniform, lwps::matrix>
		operator()(const Domain& domain,
				const order<N>& correction = default_order) const
		{
			return operator()(domain, std::get<0>(dimensions(domain)), correction);
		}

		constexpr __identity_functor () {}
	} identity;*/
}
