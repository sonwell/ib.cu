#pragma once
#include <cstddef>
#include <utility>
#include <array>
#include <type_traits>

namespace fd {
namespace boundary {
namespace __1 {

class boundary {
private:
	using container = std::array<double, 2>;
	container _params;
protected:
	constexpr boundary(const container& p) : _params{p} {}
public:
	constexpr const container& params() const { return _params; }
};

template <bool is_lower>
struct tag : std::integral_constant<bool, is_lower> {};

inline constexpr tag<true> lower;
inline constexpr tag<false> upper;


struct robin : boundary {
	static constexpr auto solid = true;
	constexpr robin(double a, double b) : boundary{{a, b}} {}
};

struct periodic : boundary {
	static constexpr auto solid = false;
	constexpr periodic() : boundary{{0, 0}} {}
};

struct dirichlet : robin { constexpr dirichlet() : robin(1, 0) {} };
struct neumann : robin { constexpr neumann() : robin(0, 1) {} };

} // namespace __1

using __1::robin;
using __1::dirichlet;
using __1::neumann;
using __1::periodic;
using __1::tag;
using __1::lower;
using __1::upper;

} // namespace boundary

template <typename T> struct is_boundary :
	std::integral_constant<bool, std::is_base_of_v<boundary::__1::boundary, T>> {};
template <typename T> inline constexpr bool is_boundary_v =
	is_boundary<T>::value;

namespace boundary {

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

} // namespace boundary
} // namespace fd
