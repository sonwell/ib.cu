#include "util/memory_resource.h"
#include "cuda/device.h"
#include "fd/grid.h"
#include "fd/boundary.h"
#include "fd/dimension.h"
#include "fd/domain.h"
#include "fd/boundary_ops.h"
#include "fd/discretization.h"
#include "fd/size.h"
#include "bases/types.h"
#include "bases/phs.h"
#include "bases/polynomials.h"
#include "bases/geometry.h"
#include "ib/sweep.h"
#include "ib/spread.h"
#include "ib/interpolate.h"
#include "forces/bending.h"
#include "forces/skalak.h"
#include "forces/combine.h"
#include "ins/solver.h"
#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

template <typename type>
struct pair {
	type data;
	type sample;
};

template <typename reference_type>
struct base_container {
private:
	using operator_type = bases::operators<2>;
	using geometry_type = bases::geometry<2>;
	using operator_pair_type = pair<const operator_type&>;
	using geometry_pair_type = pair<const geometry_type&>;
	using reference_tag = decltype(bases::reference);
	using current_tag = decltype(bases::current);
public:
	decltype(auto)
	operators() const
	{
		return operator_pair_type {
			ref.data_to_data,
			ref.data_to_sample
		};
	}

	decltype(auto)
	geometry(const reference_tag&) const
	{
		return geometry_pair_type {
			ref.data_geometry,
			ref.sample_geometry
		};
	}

	decltype(auto)
	geometry(const current_tag&) const
	{
		return geometry_pair_type {
			data,
			sample
		};
	}

	geometry_type data;
	geometry_type sample;
	const reference_type& ref;

	base_container(const reference_type& ref, const matrix& x) :
		data{ref.data_to_data, x},
		sample{ref.data_to_sample, x},
		ref{ref} {}
};

template <typename reference_type>
struct container : base_container<reference_type> {
private:
	using base = base_container<reference_type>;

	matrix&
	get_x()
	{
		return base::data.position;
	}

	void
	set_x(const matrix& x)
	{
		const auto& ops = base::operators();
		base::data = {ops.data, x};
		base::sample = {ops.sample, x};
	}

	struct constructor_info {
		int n;
		bases::geometry<2> data;
		bases::geometry<2> sample;
	};

	static decltype(auto)
	construct(const reference_type& reference, int n)
	{
		auto& y = reference.data_geometry.position;
		auto m = y.rows();
		matrix x{m, n * 3};

		auto* xdata = x.values();
		auto* ydata = y.values();
		auto k = [=] __device__ (int tid)
		{
			auto j = tid % m;
			for (int i = 0; i < 3; ++i)
				xdata[i * n * m + tid] = ydata[i * m + j];
		};
		util::transform<128, 3>(k, n * m);
		return x;
	}

	container(const reference_type& ref, int n, matrix x) :
		base{ref, std::move(x)}, n(n),
		x{[&] () -> matrix& { return get_x(); },
		  [&] (const matrix& x) { set_x(x); }} {}
public:
	int size() const { return n; }

	container(const reference_type& reference, int n) :
		container(reference, n, construct(reference, n)) {}
private:
	int n;
public:
	util::getset<matrix&> x;
};

template <typename tag_type, typename domain_type>
decltype(auto)
zeros(const tag_type& tag, const domain_type& domain)
{
	using namespace util::functional;
	auto k = [&] (const auto& comp) {
		fd::grid g{tag, domain, comp};
		return algo::vector{ fd::size(g), linalg::zero };
	};
	return map(k, fd::components(domain));
}

int
main(int argc, char** argv)
{
	int iterations;
	if (argc == 1)
		iterations = 10;
	else
		iterations = atoi(argv[1]);
	util::set_default_resource(cuda::default_device().memory());

	constexpr fd::dimension x{50_um, fd::boundary::periodic()};
	constexpr fd::dimension y{50_um, fd::boundary::dirichlet()};
	constexpr fd::dimension z{50_um, fd::boundary::periodic()};
	constexpr fd::domain domain{x, y, z};
	constexpr fd::mac mac{32};

	constexpr auto h = domain.unit() / mac.refinement();
	constexpr ins::parameters params {/*4.096 * (double) (h * h)*/ 0.1 * (double) h, 1_g / 1_mL, 1_cP, 1e-8};

	util::logging::info("characteristic length: ", domain.unit());
	util::logging::info("timestep: ", params.timestep);
	util::logging::info("viscosity: ", params.viscosity);
	util::logging::info("density: ", params.density);
	util::logging::info("diffusivity: ", params.viscosity / params.density);
	util::logging::info("lambda: ", params.timestep * params.viscosity / (params.density * domain.unit() * domain.unit()));
	util::logging::info("top wall velocity: ", 5_cm / 1_s, "k̂");

	constexpr ib::spread spread{mac, domain};
	constexpr ib::interpolate interpolate{mac, domain};

	constexpr bases::polyharmonic_spline<7> basic;
	rbc cell_ref{625, 3200, basic};
	container cells{cell_ref, 1};
	const matrix& cx = cells.x;
	auto n = cx.rows() * cx.cols() / domain.dimensions;
	double k = params.timestep;

	matrix f_c;
	auto f = [&] (const auto& v)
	{
		constexpr forces::skalak skalak{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
		constexpr forces::bending bending{2e-12_erg};
		constexpr forces::combine forces{skalak, bending};
		auto& x = cells.geometry(bases::current).data.position;
		auto n = x.rows() * x.cols() / domain.dimensions;
		auto u = interpolate(n, x, v);

		base_container tmp{cell_ref, x + k * std::move(u)};
		auto& y = tmp.geometry(bases::current).sample.position;
		auto m = y.rows() * y.cols() / domain.dimensions;
		auto f = forces(tmp);
		f_c = f;
		return spread(m, y, f);
	};

	ins::solver step{mac, domain, params};

	auto u = zeros(mac, domain);
	auto ub = zeros(mac, domain);

	using fd::correction::second_order;
	constexpr fd::grid g{mac, domain, z};
	auto b = fd::upper_boundary(g, y, second_order);
	std::get<2>(ub) = b * vector{fd::size(g, y), algo::fill((double) (5_cm/1_s))};
	std::get<2>(u) = vector{fd::size(g), algo::fill((double) (5_cm/1_s))};

	std::cout << "import numpy as np\n";
	std::cout << "import matplotlib\n";
	std::cout << "import matplotlib.pyplot as plt\n";
	std::cout << "from mpl_toolkits.mplot3d import Axes3D\n";
	std::cout << "rdbu = matplotlib.cm.get_cmap('RdBu')\n";
	std::cout << "i = np.linspace(0, 31, 32)\n"
		<< "yg, zg, xg = np.meshgrid(i + 0.5, i, i + 0.5)\n"
		<< "xg = xg.flatten() * 5 / 32\n"
		<< "yg = yg.flatten() * 5 / 32\n"
		<< "zg = zg.flatten() * 5 / 32\n";
	{
		const matrix& y = cells.geometry(bases::current).sample.position;
		std::cout << "y0 = " << linalg::io::numpy << y << std::endl;
	}

	for (int i = 0; i < iterations; ++i) {
		util::logging::info("simulation time: ", i * params.timestep);
		u = step(u, ub, f);
		auto v = k * interpolate(n, cells.x, u);
		cells.x += v;

		if (!(i % 100)) {
			//auto& sigma = cells.geometry(bases::current).sample.sigma;
			const matrix& y = cells.geometry(bases::current).sample.position;
			//std::cout << "σ = " << linalg::io::numpy << sigma << std::endl;
			std::cout << "y = " << linalg::io::numpy << y << std::endl;
			std::cout << "f = " << linalg::io::numpy << f_c << "\n";
			std::cout << "fm = np.sqrt(f[:, 0]**2 + f[:, 1]**2 + f[:, 2]**2)\n";
			std::cout << "w = " << linalg::io::numpy << std::get<2>(u) << "\n";

			std::cout << "fig = plt.figure()\n";
			std::cout << "ax = fig.add_subplot(111, projection='3d', azim=0, elev=0)\n";
			//std::cout << "ax.scatter(y[:, 0] % 5, y[:, 2] % 5, y[:, 1] % 5, c=fm)  #σ)\n";
			std::cout << "ax.scatter(y[:, 0], y[:, 2], y[:, 1], c=fm)  #σ)\n";
			/*std::cout
				<< "diff = w - 0.1 * yg\n"
				<< "max = np.max(np.abs(diff))\n"
				<< "norm = matplotlib.colors.Normalize(vmin=-max, vmax=+max)\n"
				<< "colors = rdbu(norm(diff))\n"
				<< "alpha = np.abs(diff)\n"
				<< "alpha = alpha / np.max(alpha)\n"
				<< "color = np.array([[r, g, b, a] for (r, g, b, _), a in zip(colors, alpha)])\n"
				<< "ax.scatter(xg, zg, yg, c=color)\n";*/
			std::cout << "plt.show()\n";
		}
	}

	return 0;
}
