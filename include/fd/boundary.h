#pragma once
#include <cstddef>
#include <utility>
#include <type_traits>
#include "util/array.h"

namespace fd {
namespace boundary {
namespace impl {

class base {
private:
	using container = util::array<double, 2>;
	container _params;
protected:
	constexpr base(const container& p) : _params{p} {}
public:
	constexpr const container& params() const { return _params; }
};

inline constexpr struct lower_tag : std::integral_constant<std::size_t, 1> {} lower;
inline constexpr struct upper_tag : std::integral_constant<std::size_t, 0> {} upper;

} // namespace impl

struct robin : impl::base {
	static constexpr auto solid = true;
	constexpr robin(double a, double b) : base{{a, b}} {}
};

struct periodic : impl::base {
	static constexpr auto solid = false;
	constexpr periodic() : base{{0, 0}} {}
};

struct dirichlet : robin { constexpr dirichlet() : robin(1, 0) {} };
struct neumann : robin { constexpr neumann() : robin(0, 1) {} };

template <typename T> struct is_boundary :
	std::integral_constant<bool, std::is_base_of_v<robin, T> ||
	                             std::is_base_of_v<periodic, T>> {};
template <typename T> inline constexpr bool is_boundary_v =
	is_boundary<T>::value;

template <typename lower, typename upper>
struct is_valid_combination {
private:
	static constexpr bool lower_is_solid = lower::solid;
	static constexpr bool upper_is_solid = upper::solid;
	static_assert(is_boundary_v<lower>, "invalid boundary type");
	static_assert(is_boundary_v<upper>, "invalid boundary type");
public:
	static constexpr bool value = !(lower_is_solid ^ upper_is_solid);
};

template <typename L, typename U>
inline constexpr bool is_valid_combination_v =
	is_valid_combination<L, U>::value;

using impl::lower;
using impl::upper;

} // namespace boundary

using boundary::is_boundary;
using boundary::is_boundary_v;

} // namespace fd
