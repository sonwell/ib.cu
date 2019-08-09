#include <utility>
#include <cmath>

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"
#include "fd/grid.h"
#include "fd/size.h"

#include "ins/diffusion.h"

#include "algo/types.h"
#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "algo/preconditioner.h"

#include "util/functional.h"
#include "util/debug.h"

#include "cuda/event.h"
#include "units.h"

struct chebyshev : algo::chebyshev, algo::preconditioner {
private:
	chebyshev(std::pair<double, double> range, const algo::matrix& m) :
		algo::chebyshev(std::get<1>(range), std::get<0>(range), m) {}
public:
	virtual algo::vector
	operator()(const algo::vector& b) const
	{
		return algo::chebyshev::operator()(b);
	}

	chebyshev(const algo::matrix& m) :
		chebyshev(algo::gershgorin(m), m) {}
};

template <typename grid_type>
algo::vector
fill(const grid_type& grid)
{
	static constexpr auto pi = M_PI;
	auto s = fd::size(grid);
	algo::vector r{s};

	auto* rdata = r.values();
	auto k = [=] __device__ (int tid)
	{
		auto x = grid.point(tid);
		rdata[tid] = sin(2 * pi * x[0] / 10_um);
	};
	util::transform<128, 7>(k, s);
	return r;

	//auto k = [=] __device__ (int i) { return f(grid.point(i)); };
	//return algo::vector{s, linalg::fill(k)};
}

int
main(void)
{
	static constexpr auto n = 128;
	static constexpr auto k = 1e-4_s / ((n / 128) * (n / 128));

	auto density = 1_g / (1_cm * 1_cm * 1_cm);
	auto viscosity = 0.89_cP;
	ins::simulation params{k, viscosity / density, 1e-8};
	double mu = params.coefficient;

	util::set_default_resource(cuda::default_device().memory());

	util::file_logfile logfile("diffusion.log");
	util::logger logger(logfile, util::log_level::info);
	util::set_logger(logger);

	constexpr fd::dimension x{10_um, fd::boundary::periodic()};
	constexpr fd::dimension y{10_um, fd::boundary::periodic()};
	constexpr fd::dimension z{10_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac(16);

	auto pc = [] (const auto& /*grid*/, algo::matrix& m) { return new chebyshev(m); };

	constexpr fd::grid grid{mac, domain, x};
	ins::diffusion step{grid, params, pc};
	auto u = fill(grid);
	auto ub = 0 * u;
	auto b = -mu * (fd::laplacian(grid) * u);

	for (int i = 0; i < 2; ++i)
		std::cout << (u = step(1.0, u, ub, b)) << std::endl;

	return 0;
}
