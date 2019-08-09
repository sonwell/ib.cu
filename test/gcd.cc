#include <iostream>
#include "algo/gcd.h"
#include "units.h"

int
main(void)
{
	using algo::gcd;
	constexpr auto x = 100_um;
	constexpr auto y = 50_um;
	constexpr auto z = 100_um;
	constexpr auto r = gcd(gcd(x, y), z);

	std::cout << x << ' ' << y << ' ' << z << std::endl;
	std::cout << "gcd: " << r << std::endl;
	return 0;
}
