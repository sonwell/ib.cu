#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/domain.h"
#include "fd/identity.h"
#include "fd/laplacian.h"
#include "fd/average.h"
#include "fd/differential.h"
#include "fd/size.h"
#include "fd/boundary_ops.h"
#include "fd/discretization.h"
#include "fd/grid.h"
#include "ib/sweep.h"
#include "ib/spread.h"
#include "ib/interpolate.h"
#include "bases/shapes/sphere.h"
#include "units.h"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

int
main(void)
{
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{10_um, fd::boundary::periodic()};
	constexpr fd::dimension y{10_um, fd::boundary::periodic()};
	constexpr fd::dimension z{10_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{16};

	constexpr ib::spread spread{mac, domain};
	constexpr ib::interpolate interpolate{mac, domain};

	constexpr auto n = 2;
	using bases::shapes::sphere;
	auto p = sphere::shape(sphere::sample(n));

	auto zz = 5_um;

	auto* pdata = p.values();
	auto k = [=] __device__ (int tid)
	{
		for (int i = 0; i < 3; ++i)
			pdata[n * i + tid] = zz; // + 3.91_um * pdata[n * i + tid];
	};

	auto s = 16 * 16 * 16;
	std::tuple u{ib::vector(s, linalg::one), ib::vector(s, linalg::one), ib::vector(s, linalg::one)};
	util::transform<128, 7>(k, n);

	ib::matrix f(n, 3, linalg::one);

	auto v = spread(n, p, f);
	auto w = interpolate(n, p, u);

	std::cout << linalg::io::python << std::get<0>(v) << std::endl;
	std::cout << linalg::io::python << w << std::endl;

	return 0;
}
