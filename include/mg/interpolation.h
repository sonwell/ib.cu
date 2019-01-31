#pragma once
#include <tuple>
#include <utility>
#include "util/launch.h"
#include "util/functional.h"
#include "util/debug.h"
#include "lwps/matrix.h"
#include "lwps/kron.h"
#include "fd/grid.h"
#include "fd/domain.h"
#include "fd/operators.h"
#include "fd/boundary_corrector.h"

namespace mg {
	namespace impl {
		struct interpolation_tag {};
		struct restriction_tag {};

		template <typename> struct weights;

		template <> struct weights<fd::grid::center> {
			static constexpr int size() { return 4; }

			constexpr __host__ __device__ double
			operator[](int n) const
			{
				constexpr double value[] = {0.25, 0.75, 0.75, 0.25};
				return value[n];
			}

			constexpr __host__ __device__ double
			upper_weight() const { return 0.25; }

			constexpr __host__ __device__ double
			lower_weight() const { return 0.25; }
		};

		template <> struct weights<fd::grid::edge> {
			static constexpr int size() { return 3; }

			constexpr __host__ __device__ double
			operator[](int n) const
			{
				constexpr double value[] = {0.5, 1.0, 0.5};
				return value[n];
			}

			constexpr __host__ __device__ double
			upper_weight() const { return 0.5; }

			constexpr __host__ __device__ double
			lower_weight() const { return 0.5; }
		};

		template <typename collocation_type, typename view_type>
		lwps::matrix
		interpolater(const view_type& view)
		{
			using weight_type = impl::weights<collocation_type>;
			using lower_boundary_type = typename view_type::lower_boundary_type;
			using upper_boundary_type = typename view_type::upper_boundary_type;
			using corrector_type = fd::boundary::corrector<
				collocation_type, lower_boundary_type, upper_boundary_type>;
			static constexpr auto nw = weight_type::size();
			static constexpr auto on_boundary = collocation_type::on_boundary;
			static constexpr auto solid_boundary = view_type::solid_boundary;
			static constexpr auto correction = solid_boundary && on_boundary;
			static constexpr auto upper_missing = !correction;
			static constexpr auto lower_missing = !on_boundary;
			static constexpr auto missing = upper_missing + lower_missing;

			const auto gridpts = view.gridpts();
			const auto rows = gridpts - correction;
			const auto cols = (gridpts >> 1) - correction;
			const auto nonzero = nw * cols - missing;
			corrector_type corrector(view);
			weight_type weights;

			lwps::matrix result{rows, cols, nonzero};
			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();
			auto k = [=] __device__ (int tid)
			{
				constexpr auto fix = correction ? nw - nw/2 : nw/2;
				if (tid < rows) starts[tid] = tid ? nw * (tid / 2) + fix * (tid & 1) - fix/2: 0;
				auto shift = (tid + upper_missing) % nw;
				auto col = (tid + upper_missing) / nw;
				/* XXX hard coded for 4 and 3 weights, respectively */
				if constexpr(std::is_same_v<collocation_type, fd::grid::center>) {
					constexpr int swizzle[] = {3, 1, 2, 0};
					constexpr int shifts[] = {-1, 0, 0, 1};
					auto value = weights[swizzle[shift]];
					indices[tid] = col + shifts[shift] + lwps::indexing_base;
					values[tid] = value;
				} else {
					auto value = weights[shift];
					indices[tid] = col + lwps::indexing_base;
					values[tid] = value;
				}
				if (!tid) starts[rows] = nonzero;
			};
			util::transform<128, 7>(k, nonzero);
			result += corrector.upper_stencil(rows, cols, weights.upper_weight());
			if constexpr(solid_boundary || !on_boundary)
				result += corrector.lower_stencil(rows, cols, weights.lower_weight());
			return std::move(result);
		}

		template <typename collocation_type, typename view_type>
		lwps::matrix
		restricter(const view_type& view)
		{
			using weight_type = impl::weights<collocation_type>;
			using lower_boundary_type = typename view_type::lower_boundary_type;
			using upper_boundary_type = typename view_type::upper_boundary_type;
			using corrector_type = fd::boundary::corrector<
				collocation_type, lower_boundary_type, upper_boundary_type>;
			static constexpr auto nw = weight_type::size();
			static constexpr auto on_boundary = collocation_type::on_boundary;
			static constexpr auto solid_boundary = view_type::solid_boundary;
			static constexpr auto correction = solid_boundary && on_boundary;
			static constexpr auto upper_missing = !correction;
			static constexpr auto lower_missing = !on_boundary;
			static constexpr auto missing = upper_missing + lower_missing;

			const auto gridpts = view.gridpts();
			const auto rows = (gridpts >> 1) - correction;
			const auto cols = gridpts - correction;
			const auto nonzero = nw * rows - missing;
			corrector_type corrector(view);
			weight_type weights;

			lwps::matrix result{rows, cols, nonzero};
			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();
			auto k = [=] __device__ (int tid)
			{
				if (tid < rows) starts[tid] = tid ? nw * tid - upper_missing : 0;
				auto shift = (tid + upper_missing) % nw;
				auto value = weights[shift];
				auto row = (tid + upper_missing) / nw;
				auto col = 2 * row + shift - upper_missing;
				indices[tid] = col + lwps::indexing_base;
				values[tid] = value / 2;
				if (!tid) starts[rows] = nonzero;
			};
			util::transform<128, 7>(k, nonzero);
			result += corrector.lower_stencil(rows, cols, weights.lower_weight() / 2);
			if constexpr(solid_boundary || !on_boundary)
				result += corrector.upper_stencil(rows, cols, weights.upper_weight() / 2);
			return std::move(result);
		}

		template <typename grid_type>
		class builder {
			private:
				using sequence = std::make_index_sequence<std::tuple_size<grid_type>::value>;
				template <std::size_t N> using collocation = std::tuple_element_t<N, grid_type>;

				template <typename collocation_type, typename view_type>
				static inline auto
				one(const view_type& view, const interpolation_tag&)
				{
					return interpolater<collocation_type>(view);
				}

				template <typename collocation_type, typename view_type>
				static inline auto
				one(const view_type& view, const restriction_tag&)
				{
					return restricter<collocation_type>(view);
				}

				template <std::size_t ... ns, typename view_types, typename tag_type>
				static auto
				splat(const std::index_sequence<ns...>&, const view_types& views, const tag_type& tag)
				{
					return std::make_tuple(one<collocation<ns>>(std::get<ns>(views), tag)...);
				}
			public:
				template <typename view_types, typename tag_type>
				static auto
				build(const view_types& views, const tag_type& tag)
				{
					using namespace util::functional;

					auto&& multikron = partial(foldl, lwps::kron);
					auto&& components = splat(sequence(), views, tag);
					auto&& reversed = reverse(std::move(components));
					return apply(multikron, std::move(reversed));
				}
		};
	}

	template <typename domain_type, typename view_type>
	decltype(auto)
	interpolation(const domain_type& domain, const view_type& view)
	{
		using fd::operators::caller;
		using impl::builder;
		using impl::interpolation_tag;
		using tag_type = typename domain_type::tag_type;
		using caller_type = caller<builder, tag_type, 0, domain_type::ndim>;
		auto&& views = fd::dimensions(domain);
		return caller_type::call(view, views, interpolation_tag());
	}

	template <typename domain_type, typename view_type>
	decltype(auto)
	restriction(const domain_type& domain, const view_type& view)
	{
		using fd::operators::caller;
		using impl::builder;
		using impl::restriction_tag;
		using tag_type = typename domain_type::tag_type;
		using caller_type = caller<builder, tag_type, 0, domain_type::ndim>;
		auto&& views = fd::dimensions(domain);
		return caller_type::call(view, views, restriction_tag());
	}
}