#include <utility>
#include <cmath>

#include "lwps/vector.h"
#include "lwps/matrix.h"
#include "lwps/io.h"

#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"
#include "fd/grid.h"
#include "fd/size.h"

#include "ins/diffusion.h"

#include "algo/chebyshev.h"
#include "algo/gershgorin.h"
#include "algo/preconditioner.h"

#include "util/functional.h"
#include "util/debug.h"

#include "cuda/event.h"
#include "units.h"

static constexpr auto pi = M_PI;

struct chebyshev : algo::chebyshev, algo::preconditioner {
private:
	chebyshev(std::pair<double, double> range, const lwps::matrix& m) :
		algo::chebyshev(std::get<1>(range), std::get<0>(range), m) {}
public:
	virtual lwps::vector
	operator()(const lwps::vector& b) const
	{
		return algo::chebyshev::operator()(b);
	}

	chebyshev(const lwps::matrix& m) :
		chebyshev(algo::gershgorin(m), m) {}
};

template <typename domain_type, typename ... initializer_types>
auto
init(const domain_type& domain, initializer_types ... initializers)
{
	using namespace util::functional;
	static constexpr auto dimensions = domain_type::ndim;
	static_assert(sizeof...(initializer_types) == dimensions,
			"number of initializers must match the domain dimension");

	auto k = [&] (const auto& view, auto initializer)
	{
		using view_type = decltype(view);
		auto n = domain.resolution();
		auto size = fd::size(domain, view);
		lwps::vector u(size);
		auto* values = u.values();

		auto f = [=] __device__ (int tid, auto f)
		{
			double x[dimensions];
			int index = tid;
			for (int i = 0; i < dimensions; ++i) {
				x[i] = ((double) (index % n)) / n;
				index /= n;
			}
			values[tid] = f(x);
		};
		util::transform<128, 7>(f, size, initializer);
		return std::move(u);
	};
	const auto& views = fd::dimensions(domain);
	auto&& inits = std::make_tuple(initializers...);
	auto&& args = zip(views, inits);
	auto&& u = map(partial(apply, k), args);
	return std::move(u);
}

int
main(void)
{
	static constexpr auto n = 128;
	static constexpr auto k = 1e-4_s / ((n / 128) * (n / 128));

	auto density = 1_g / (1_cm * 1_cm * 1_cm);
	auto viscosity = 0.89_cP;
	ins::simulation params{k, viscosity / density, 1e-8};
	lwps::io::set_style(lwps::io::style::python());

	util::file_logfile logfile("diffusion.log");
	util::logger logger(logfile, util::log_level::info);
	util::set_logger(logger);

	constexpr fd::dimension x{10_um, fd::boundary::periodic()};
	constexpr fd::dimension y{10_um, fd::boundary::periodic()};
	constexpr fd::dimension z{10_um, fd::boundary::periodic()};
	constexpr fd::domain domain{fd::grid::mac(n), x, y, z};
	static constexpr auto dimensions = decltype(domain)::ndim;

	auto pc = [] (const auto&, const auto&, const lwps::matrix& m) { return new chebyshev(m); };

	ins::diffusion step(domain, x, params, pc);
	auto&& u = init(domain,
		[] __device__ (double (&x)[dimensions]) { return sin(10 * pi * x[0]); },
		[] __device__ (double (&x)[dimensions]) { return sin(10 * pi * x[1]); },
		[] __device__ (double (&x)[dimensions]) { return sin(10 * pi * x[2]); }
	);
	auto&& ub = init(domain,
		[] __device__ (double (&x)[dimensions]) { return 0; },
		[] __device__ (double (&x)[dimensions]) { return 0; },
		[] __device__ (double (&x)[dimensions]) { return 0; }
	);
	auto&& b = init(domain,
		[] __device__ (double (&x)[dimensions]) { return 1; }, //100 * pi * pi * sin(10 * pi * x[0]); },
		[] __device__ (double (&x)[dimensions]) { return 0; }, //sin(16 * pi * x[1]); },
		[] __device__ (double (&x)[dimensions]) { return 0; } //sin(16 * pi * x[2]); }
	);

	auto& v = u[0];
	auto& vb = ub[0];
	auto& c = b[0];

	auto* style = lwps::io::get_style();
	for (int i = 0; i < 1; ++i) {
		v = step(1.0, v, vb, c);
	}

	return 0;
}
