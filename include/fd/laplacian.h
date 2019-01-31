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
	namespace laplacian_impl {
		template <typename Collocation, typename View>
		lwps::matrix
		laplacian(const View& view)
		{
			using lower_boundary_type = typename View::lower_boundary_type;
			using upper_boundary_type = typename View::upper_boundary_type;
			using corrector_type = boundary::corrector<Collocation,
				  lower_boundary_type, upper_boundary_type>;
			corrector_type corrector(view);
			const auto rows = corrector.gridpts();
			const auto n = view.resolution();
			if (rows == 0) return lwps::matrix{0, 0};

			auto nonzero = 3 * rows - 2;
			lwps::matrix result{rows, rows, nonzero};
			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();
			const auto scale = n * n;

			auto k = [=] __device__ (int tid)
			{
				if (tid < rows) starts[tid] = tid ? 3 * tid - 1 : 0;
				auto loc = (tid + 1) % 3;
				auto row = (tid + 1) / 3;
				auto col = loc + row - 1;
				indices[tid] = col + lwps::indexing_base;
				values[tid] = (1 - 3 * (loc & 1)) * scale;
				if (!tid) starts[rows] = nonzero;
			};
			util::transform<128, 7>(k, nonzero);

			result += corrector.lower_stencil(rows, rows, scale)
			        + corrector.upper_stencil(rows, rows, scale);
			return std::move(result);
		}

		using identity_impl::identity;

		template <typename Grid>
		class builder {
		private:
			using grid_type = Grid;
			using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
			template <std::size_t N> using collocation = std::tuple_element_t<N, grid_type>;
			static constexpr auto& order = fd::boundary::correction::second_order;

			template <typename collocation_type, typename view_type>
			static auto
			one(const view_type& view)
			{
				auto&& lap = laplacian<collocation_type>(view);
				auto&& id = identity<collocation_type>(view, order);
				return std::pair{std::move(lap), std::move(id)};
			}

			template <std::size_t ... Ns, typename Views>
			static auto
			splat(const std::index_sequence<Ns...>&, const Views& views)
			{
				return std::make_tuple(one<collocation<Ns>>(std::get<Ns>(views))...);
			}
		public:
			template <typename Views>
			static lwps::matrix
			build(const Views& views)
			{
				using namespace util::functional;

				static auto operation = [] (const auto& left, const auto& right)
				{
					using lwps::kron;
					auto&& l_lap = left.first;
					auto&& l_id = left.second;
					auto&& r_lap = right.first;
					auto&& r_id = right.second;
					return std::pair{kron(l_lap, r_id) + kron(l_id, r_lap), kron(l_id, r_id)};
				};

				auto&& pairs = splat(sequence(), views);
				auto&& multiop = partial(foldl, operation);
				auto&& reversed = reverse(std::move(pairs));
				auto&& results = apply(multiop, std::move(reversed));
				return std::move(results.first);
			}
		};
	}

	template <typename domain_type, typename view_type>
	auto
	laplacian(const domain_type& domain, const view_type& view)
	{
		using operators::caller;
		using laplacian_impl::builder;
		using tag_type = typename domain_type::tag_type;
		static constexpr auto dimensions = domain_type::ndim;
		using caller_type = caller<builder, tag_type, 0, dimensions>;
		auto&& views = fd::dimensions(domain);
		return caller_type::call(view, views);
	}

	template <typename domain_type>
	auto
	laplacian(const domain_type& domain)
	{
		static_assert(grid::is_uniform_v<domain_type::tag_type>,
				"the 1-argument variant of fd::laplacian requires a uniform grid (cell- or vertex-centered)");
		return laplacian(domain, std::get<0>(dimensions(domain)));
	}

	/*static constexpr struct __laplacian_functor {
		template <typename Domain, typename View>
		lwps::matrix
		operator()(const Domain& domain, const View& view) const
		{
			using operators::caller;
			using laplacian_impl::builder;
			using tag_type = typename Domain::tag_type;
			using caller_type = caller<builder, tag_type, 0, Domain::ndim>;
			auto&& views = fd::dimensions(domain);
			return caller_type::call(view, views);
		}

		template <typename Domain>
		typename std::enable_if<Domain::tag_type::is_uniform, lwps::matrix>::type
		operator()(const Domain& domain) const
		{
			return operator()(domain, std::get<0>(fd::dimensions(domain)));
		}

		constexpr __laplacian_functor () {}
	} laplacian;*/
}
