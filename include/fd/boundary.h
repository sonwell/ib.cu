#pragma once
#include <cstddef>
#include <utility>
#include <type_traits>

namespace fd {
	namespace boundary_impl {
		template <typename> class params_container;

		template <std::size_t ... N>
		class params_container<std::index_sequence<N...>> {
			public:
				static constexpr std::size_t order = sizeof...(N);
				using container_type = double[order];
			private:
				container_type _params;
			protected:
				constexpr params_container(const container_type& params) :
					_params{params[N]...} {}
			public:
				constexpr const container_type& params() const { return _params; }
		};

		template <std::size_t N>
		using base_boundary = params_container<std::make_index_sequence<N>>;

		class robin : public base_boundary<2ull> {
			private:
				using base = base_boundary<2ull>;
			public:
				static constexpr std::size_t width = 1;
				static constexpr char lower_repr = 'x';
				static constexpr char upper_repr = 'x';

				constexpr robin(double a, double b) : base{{a, b}} {}
		};

		class periodic {
			public:
				using container_type = double[2];
				static constexpr std::size_t width = 0;
				static constexpr char lower_repr = '<';
				static constexpr char upper_repr = '>';
			private:
				static constexpr container_type _params = {0, 0};
			public:
				constexpr const container_type& params() const { return _params; }
				constexpr periodic() {}
		};

		class dirichlet : public robin {
			public:
				static constexpr char lower_repr = '|';
				static constexpr char upper_repr = '|';

				constexpr dirichlet() : robin(1, 0) {}
		};

		class neumann : public robin {
			public:
				static constexpr char lower_repr = '(';
				static constexpr char upper_repr = ')';

				constexpr neumann() : robin(0, 1) {}
		};

		template <typename T> struct is_boundary_type :
			std::integral_constant<bool, std::is_base_of<robin, T>::value ||
			                             std::is_base_of<periodic, T>::value> {};
		template <typename T> inline constexpr bool is_boundary_type_v =
			is_boundary_type<T>::value;

		template <typename Lower, typename Upper>
		struct is_valid_combination : std::true_type {
			static_assert(is_boundary_type_v<Lower>, "invalid boundary type");
			static_assert(is_boundary_type_v<Upper>, "invalid boundary type");
		};
		template <> struct is_valid_combination<periodic, periodic> : std::true_type {};
		template <typename Lower>
		struct is_valid_combination<Lower, periodic> : std::false_type {
			static_assert(is_boundary_type_v<Lower>, "invalid boundary type");
		};
		template <typename Upper>
		struct is_valid_combination<periodic, Upper> : std::false_type {
			static_assert(is_boundary_type_v<Upper>, "invalid boundary type");
		};

		template <typename L, typename U> inline constexpr bool is_valid_combination_v =
			is_valid_combination<L, U>::value;

		static constexpr struct lower_tag : std::integral_constant<std::size_t, 1> {} lower;
		static constexpr struct upper_tag : std::integral_constant<std::size_t, 0> {} upper;
	}

	namespace boundary {
		using boundary_impl::robin;
		using boundary_impl::periodic;
		using boundary_impl::dirichlet;
		using boundary_impl::neumann;
		using boundary_impl::is_boundary_type;
		using boundary_impl::is_boundary_type_v;
		using boundary_impl::is_valid_combination;
		using boundary_impl::is_valid_combination_v;
	}
}
