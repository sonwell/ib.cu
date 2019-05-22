#include <cmath>
#include <utility>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include "cuda/event.h"
#include "algo/types.h"
#include "algo/pcg.h"
#include "algo/redblack.h"
#include "algo/symmilu.h"
#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "fd/domain.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/laplacian.h"
#include "util/launch.h"

class chebyshev : public algo::chebyshev, public algo::preconditioner {
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

template <typename domain_type>
algo::symmilu
redblack_ilu(const domain_type& domain, const algo::matrix& m)
{
	const auto& views = fd::dimensions(domain);
	const auto& view = std::get<0>(views);
	auto* rb = new algo::redblack(domain, view);
	return algo::symmilu(m, rb);
}

int
main(void)
{
	static constexpr auto n = 128;
	static constexpr auto pi = M_PI;
	constexpr fd::dimension x{100_um, fd::boundary::periodic()};
	constexpr fd::dimension y{100_um, fd::boundary::dirichlet()};
	constexpr fd::dimension z{100_um, fd::boundary::periodic()};
	constexpr fd::domain domain{fd::grid::mac(n), x};
	static constexpr auto dimensions = decltype(domain)::ndim;

	auto k = 1e-6 * (128 * 128) / (n * n);
	auto&& id = fd::identity(domain, x);
	auto&& lap = fd::laplacian(domain, x);
	auto&& hhz = id - k / 2 * lap;

	algo::vector v(n);
	algo::vector b(n);
	auto* vvalues = v.values();
	auto* bvalues = b.values();
	auto f = [=] __device__ (int tid)
	{
		auto i = tid % n;
		auto j = (tid / n) % n;
		auto k = (tid / (n * n)) % n;
		vvalues[tid] = sin(120 * pi * i / n); // * sin(114 * pi * (j + 0.5) / n) * sin(124 * pi * (k + 0.5) / n);
		bvalues[tid] = 1.0;
	};
	util::transform(f, n);
	std::cout << b << std::endl;

	chebyshev pc{hhz};
	std::cout << pc(hhz * b) << std::endl;
	/*auto ilu = redblack_ilu(domain, hhz);

	for (int i = 0; i < 2; ++i) {
		cuda::event start, stop;
		start.record();
		auto&& q = algo::krylov::pcg(pc, hhz, k * (lap * v + b), 1e-8);
		stop.record();
		std::cout << (stop - start) << "ms" << std::endl;
	}*/

	return 0;
}
