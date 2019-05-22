#include <iostream>

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/size.h"

int
main(void)
{
	constexpr fd::dimension x(2.0_mm, fd::boundary::periodic());
	constexpr fd::dimension y(1.0_mm, fd::boundary::dirichlet());
	constexpr fd::dimension z(1.0_mm, fd::boundary::periodic());

	constexpr fd::domain domain(fd::grid::mac(128), x, y);
	auto&& views = fd::dimensions(domain);

	std::cout << std::get<0>(views).resolution() << std::endl;

	std::cout << domain.ndim << std::endl;
	std::cout << fd::size(domain, x) << std::endl;
	std::cout << (128 * 128) << std::endl;
	return 0;
}
