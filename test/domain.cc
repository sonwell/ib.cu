#include <iostream>

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/size.h"

int
main(void)
{
	constexpr fd::dimension x{2.0_mm, fd::boundary::periodic};
	constexpr fd::dimension y{1.0_mm, fd::boundary::dirichlet};
	constexpr fd::domain domain{x, y};
	constexpr fd::grid grid{fd::mac(128), domain, x};

	std::cout << "domain unit: " << domain.unit() << std::endl;
	std::cout << "grid resolution: " << grid.resolution() << " points per meter" << std::endl;
	std::cout << "domain dimensions: " << domain.dimensions << std::endl;
	std::cout << "grid points: " << fd::size(grid) << std::endl;
	std::cout << "x == y => " << std::boolalpha << (x == y) << std::endl;
	return 0;
}
