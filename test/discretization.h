#include <iostream>
#include "ib/view.h"
#include "fd/domain.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/grid.h"

int
main(void)
{
	fd::dimension x(10_m, fd::boundary::periodic());
	fd::dimension y(10_m, fd::boundary::dirichlet());
	fd::dimension z(10_m, fd::boundary::periodic());
	fd::domain domain(fd::grid::mac(4), x, y, z);
	using domain_type = std::decay_t<decltype(domain)>;
	static constexpr auto dimensions = domain_type::ndim;
	using tag_type = domain_type::tag_type;
	using grid_type = fd::grid::make<tag_type, 0, dimensions>;

	auto discr = ib::discretize<grid_type>(domain);

	auto k = [=] __device__ (int tid)
	{
		double z = tid / 32.;
		util::array x = {-5. + 20. * z, 10. * z, 10. * z};
		auto [ndx, j] = discr.ndx(x);
		auto v = discr.delta_values(ndx);
		printf("%2d %f %f %f "
				"%f %f %f %f "
				"%f %f %f %f "
				"%f %f %f %f "
				"%f %f %f %f\n",
				tid, ndx[0], ndx[1], ndx[2],
				v[ 0], v[ 1], v[ 2], v[ 3],
				v[ 4], v[ 5], v[ 6], v[ 7],
				v[ 8], v[ 9], v[10], v[11],
				v[12], v[13], v[14], v[15]);
	};
	util::transform(k, 32);
	cuda::synchronize();

	return 0;
}
