#include <iostream>
#include "algo/gcd.h"
#include "units.h"

int
main(void)
{
	using algo::gcd;
	constexpr double x = 100_um;
	constexpr double y = 50_um;
	constexpr double z = 100_um;
	constexpr auto r = gcd(gcd(x, y), z);

	std::cout << x << ' ' << y << ' ' << z << std::endl;
	std::cout << "gcd: " << r << std::endl;
	return 0;
}
