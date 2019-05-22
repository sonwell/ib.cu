#include <iostream>
#include <limits>
#include <array>
#include "units.h"
#include "util/functional.h"
#include "algo/gcd.h"

namespace impl {
constexpr int
round(double a)
{
	return (int) (a + 0.5);
}
}

struct dimension {
	units::distance _length;

	constexpr dimension(units::distance length) :
		_length(length) {}
};

struct view : dimension {
	double _resolution;
	int _gridpts;

	constexpr view(const dimension& dim, int r, double gcd) :
		dimension(dim),
		_resolution(r / gcd),
		_gridpts(r * impl::round(dim._length.value / gcd)) {}
};

template <typename ... dimensions>
struct domain {
	static constexpr auto reduce = util::functional::partial(util::functional::foldl, algo::gcd);
	std::array<view, sizeof...(dimensions)> _views;

	constexpr domain(int r, double gcd, const dimensions& ... dims) :
		_views{view(dims, r, gcd)...} {}

	constexpr domain(int r, const dimensions& ... dims) :
		domain(r, reduce(dims._length.value...), dims...) {}
};

int
main(void)
{
	constexpr dimension x{10.0_mm};
	constexpr dimension y{10.0_mm};
	constexpr domain d{16, x, y};

	auto&& views = d._views;
	auto&& view_x = std::get<0>(views);
	auto&& view_y = std::get<1>(views);

	std::cout << view_x._gridpts << ' ' << view_x._resolution << std::endl;
	std::cout << view_y._gridpts << ' ' << view_y._resolution << std::endl;

	return 0;
}
