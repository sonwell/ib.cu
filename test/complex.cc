#include <iostream>
#include "linalg/complex.h"

int
main(void)
{
	linalg::complex z(3, 1);
	std::cout << z * 3 << std::endl;

	return 0;
}
