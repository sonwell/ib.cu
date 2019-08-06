#pragma once
#include <cstddef>
#include <array>
#include <tuple>
#include <utility>
#include <type_traits>
#include "algo/gcd.h"
#include "util/functional.h"
#include "types.h"
#include "cell.h"
#include "dimension.h"

namespace fd {
namespace __1 {

using namespace util::functional;

template <typename ... dimension_types>
class domain {
	static_assert((is_dimension_v<dimension_types> && ...));
public:
	static constexpr auto dimensions = sizeof...(dimension_types);
	using container_type = std::tuple<dimension_types...>;
private:
	static constexpr auto gcd = partial(foldl, algo::gcd);
protected:
	units::distance _base_unit;
	container_type _components;
public:
	constexpr auto unit() const { return _base_unit; }
	constexpr const auto& components() const { return _components; }

	constexpr domain(const dimension_types& ... dimensions) :
		_base_unit(gcd(dimensions.length()...)),
		_components{dimensions...} {}
};

template <typename ... dimension_types>
constexpr auto
components(const domain<dimension_types...>& domain)
{
	return domain.components();
}

} // namespace __1

using __1::domain;
using __1::components;

template <typename> struct is_domain : std::false_type {};

template <typename ... dimension_types>
struct is_domain<__1::domain<dimension_types...>> :
	std::integral_constant<bool, (is_dimension_v<dimension_types> && ...)> {};

template <typename domain_type>
inline constexpr auto is_domain_v = is_domain<domain_type>::value;

}
