#pragma once
#include <cstddef>
#include "grid.h"
#include "boundary.h"
#include "dimension.h"
#include "lwps/types.h"
#include "lwps/matrix.h"
#include "util/launch.h"

namespace fd {
	namespace boundary_corrector_impl {
		template <std::size_t N> struct order :
			std::integral_constant<std::size_t, N> {};

		template <typename> struct boundary_weights;

		template <>
		struct boundary_weights<grid_impl::center> {
			lwps::value_type boundary_weight;
			lwps::value_type stencil_weight;
			lwps::value_type laplacian_correction;
			lwps::index_type nonzero;

			constexpr boundary_weights(const lwps::value_type (&p)[2], const lwps::value_type& h) :
				boundary_weight(2 * h / (h * p[0] - 2 * p[1])),
				stencil_weight(- (h * p[0] + 2 * p[1]) / (h * p[0] - 2 * p[1])),
				laplacian_correction(- h * p[0] / (4 * (h * p[0] - 2 * p[1]))),
				nonzero(stencil_weight != 0) {}
		};

		template <>
		struct boundary_weights<grid_impl::edge> {
			lwps::value_type boundary_weight;
			lwps::value_type stencil_weight;
			lwps::value_type laplacian_correction;
			lwps::index_type nonzero;

			constexpr boundary_weights(const lwps::value_type (&p)[2], const lwps::value_type& h) :
				boundary_weight(h / (h * p[0] - p[1])),
				stencil_weight(- p[1] / (h * p[0] - p[1])),
				laplacian_correction(- p[1] / (2 * (h * p[0] - p[1]))),
				nonzero(stencil_weight != 0) {}
		};

		template <typename Collocation, typename Lower, typename Upper>
		class corrector {
		private:
			using weight_type = boundary_weights<Collocation>;
			weight_type _weights[2];
			lwps::index_type _gridpts;
		public:
			using collocation_type = Collocation;
			using lower_boundary_type = Lower;
			using upper_boundary_type = Upper;
			static constexpr bool on_boundary = collocation_type::on_boundary;
			static constexpr bool solid_boundary = true;

			lwps::matrix lower_stencil(lwps::index_type, lwps::index_type, lwps::value_type = 1.0) const;
			lwps::matrix upper_stencil(lwps::index_type, lwps::index_type, lwps::value_type = 1.0) const;
			lwps::matrix stencil(lwps::index_type, lwps::index_type, lwps::value_type,
					const weight_type&, lwps::index_type, lwps::index_type) const;
			constexpr const lwps::value_type& boundary_weight(const boundary_impl::lower_tag&) const;
			constexpr const lwps::value_type& boundary_weight(const boundary_impl::upper_tag&) const;
			constexpr const lwps::index_type& gridpts() const { return _gridpts; }
			template <std::size_t N> std::tuple<double, double> identity_weights(const order<N>&) const;

			constexpr corrector(const dimension_impl::view<Lower, Upper>& view) :
				_weights{{view.lower().params(), 1.0 / view.resolution()},
						 {view.upper().params(), -1.0 / view.resolution()}},
				_gridpts(view.gridpts() - on_boundary) {}
		};

		template <typename Collocation, typename Lower, typename Upper>
		lwps::matrix
		corrector<Collocation, Lower, Upper>::stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type scale,
				const weight_type& weights, lwps::index_type fill, lwps::index_type column) const
		{
			if (scale == 0 || rows == 0 || cols == 0)
				return lwps::matrix{rows, cols};
			auto weight = weights.stencil_weight;
			auto nonzero = weights.nonzero;
			lwps::matrix result{rows, cols, nonzero, rows+1, nonzero, nonzero};

			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();

			auto k = [=] __device__ (int tid)
			{
				starts[tid] = tid ? fill : 0;

				if (!tid) {
					starts[rows] = nonzero;
					if (nonzero > 0) {
						indices[0] = column + lwps::indexing_base;
						values[0] = scale * weight;
					}
				}
			};
			util::transform<128, 7>(k, rows);
			return std::move(result);
		}

		template <typename Collocation, typename Lower, typename Upper>
		template <std::size_t N>
		std::tuple<double, double>
		corrector<Collocation, Lower, Upper>::identity_weights(const order<N>&) const
		{
			if constexpr(N == 2)
				return {_weights[0].laplacian_correction, _weights[1].laplacian_correction};
			else
				return {0, 0};
		}

		template <typename Collocation, typename Lower, typename Upper>
		inline lwps::matrix
		corrector<Collocation, Lower, Upper>::lower_stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type scale) const
		{
			auto&& weight = _weights[0];
			return stencil(rows, cols, scale, weight, weight.nonzero, 0);
		}

		template <typename Collocation, typename Lower, typename Upper>
		inline lwps::matrix
		corrector<Collocation, Lower, Upper>::upper_stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type scale) const
		{
			auto&& weight = _weights[1];
			return stencil(rows, cols, scale, weight, 0, cols-1);
		}

		template <typename Collocation, typename Lower, typename Upper>
		constexpr const double&
		corrector<Collocation, Lower, Upper>::boundary_weight(
				const boundary_impl::lower_tag&) const
		{
			return _weights[0].boundary_weight;
		}

		template <typename Collocation, typename Lower, typename Upper>
		constexpr const double&
		corrector<Collocation, Lower, Upper>::boundary_weight(
				const boundary_impl::upper_tag&) const
		{
			return _weights[1].boundary_weight;
		}

		template <typename Collocation>
		class corrector<Collocation, boundary::periodic, boundary::periodic> {
		private:
			using boundary_type = boundary::periodic;
			static constexpr lwps::value_type _boundary_weight = 0.0;
			lwps::index_type _gridpts;

			lwps::matrix stencil(lwps::index_type, lwps::index_type,
					lwps::value_type, lwps::index_type, lwps::index_type) const;
		public:
			using collocation_type = Collocation;
			using lower_boundary_type = boundary_type;
			using upper_boundary_type = boundary_type;
			static constexpr bool on_boundary = collocation_type::on_boundary;
			static constexpr bool solid_boundary = false;

			lwps::matrix lower_stencil(lwps::index_type, lwps::index_type, lwps::value_type = 1.0) const;
			lwps::matrix upper_stencil(lwps::index_type, lwps::index_type, lwps::value_type = 1.0) const;
			constexpr const lwps::value_type& boundary_weight(const boundary_impl::lower_tag&) const { return _boundary_weight; }
			constexpr const lwps::value_type& boundary_weight(const boundary_impl::upper_tag&) const { return _boundary_weight; }
			constexpr const lwps::index_type& gridpts() const { return _gridpts; }
			std::tuple<double, double> identity_weights(...) const { return {0, 0}; }

			constexpr corrector(const dimension_impl::view<boundary_type, boundary_type>& view) :
				_gridpts(view.gridpts()) {}
		};

		template <typename Collocation>
		lwps::matrix
		corrector<Collocation, boundary::periodic, boundary::periodic>::stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type weight,
				lwps::index_type fill, lwps::index_type column) const
		{
			if (weight == 0 || rows == 0 || cols == 0)
				return lwps::matrix{rows, cols};
			lwps::matrix result{rows, cols, 1};
			auto* starts = result.starts();
			auto* indices = result.indices();
			auto* values = result.values();

			auto k = [=] __device__ (int tid)
			{
				starts[tid] = tid ? fill : 0;

				if (!tid) {
					starts[rows] = 1;
					indices[0] = column + lwps::indexing_base;
					values[0] = weight;
				}
			};
			util::transform<128, 7>(k, rows);
			return std::move(result);
		}

		template <typename Collocation>
		inline lwps::matrix
		corrector<Collocation, boundary::periodic, boundary::periodic>::lower_stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type scale) const
		{
			return stencil(rows, cols, scale, 1, cols-1);
		}

		template <typename Collocation>
		inline lwps::matrix
		corrector<Collocation, boundary::periodic, boundary::periodic>::upper_stencil(
				lwps::index_type rows, lwps::index_type cols, lwps::value_type scale) const
		{
			return stencil(rows, cols, scale, 0, 0);
		}
	}

	namespace boundary {
		using boundary_corrector_impl::corrector;

		namespace correction {
			using boundary_corrector_impl::order;
			static constexpr order<0> zeroth_order;
			static constexpr order<1> first_order;
			static constexpr order<2> second_order;
		}
	}
}
