#include <iostream>
#include "algo/cross.h"

int
main(void)
{
	double e1[] = {1, 0, 0};
	double e2[] = {0, 0, 1};
	auto&& k = algo::cross(e1, e2);

	std::cout << k[0] << ' ' << k[1] << ' ' << k[2] << std::endl;
	return 0;
}
