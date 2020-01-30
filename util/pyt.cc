#include "util/functional.h"
#include "bases/types.h"
#include "bases/container.h"
#include "bases/phs.h"
#include "forces/bending.h"
#include "forces/skalak.h"
#include "forces/neohookean.h"
#include "forces/repelling.h"
#include "forces/combine.h"
#include "units.h"
#include "rbc.h"

using bases::matrix;
using bases::vector;

struct python_writer {
	std::ostream& output = std::cout;

	template <typename reference_type>
	matrix
	forces(const bases::container<reference_type>& container)
	{
		constexpr forces::skalak tension{2.5e-3_dyn/1_cm, 2.5e-1_dyn/1_cm};
		constexpr forces::bending bending{2e-12_erg};
		constexpr forces::repelling repelling{2.5e-3_dyn/1_cm};
		constexpr forces::combine forces{tension, bending, repelling};
		return forces(container);
	}

	template <typename reference_type>
	void
	initialize(const bases::container<reference_type>& container)
	{
		output << "import numpy as np\n";
		output << "from mayavi import mlab\n";
		output << "import rbc_plotter\n\n";

		auto n = container.points().sample;
		auto p = bases::traits<reference_type>::sample(n);
		const auto& [data, sample] = container.geometry(bases::current);
		auto f = forces(container);
		output << "p = " << linalg::io::numpy << p << '\n';
		output << "x0 = " << linalg::io::numpy << data.position << '\n';
		output << "y0 = " << linalg::io::numpy << sample.position << '\n';
		output << "f0 = " << linalg::io::numpy << f << '\n';
		output << "plotter = rbc_plotter.Plotter(rbc_plotter.Sphere, p, x0, y0, f0)\n";
	}


	template <typename reference_type>
	python_writer(const bases::container<reference_type>& container,
			std::ostream& output = std::cout) :
		output(output) { initialize(container); }

	~python_writer() { output << "plotter.animate()\n"; }

	template <typename reference_type>
	void
	operator()(const bases::container<reference_type>& container)
	{
		const auto& [data, sample] = container.geometry(bases::current);
		auto f = forces(container);
		output << "x = " << linalg::io::numpy << data.position << '\n';
		output << "y = " << linalg::io::numpy << sample.position << '\n';
		output << "f = " << linalg::io::numpy << f << '\n';
		output << "plotter.plot(x, y, f)\n";
	}
};

struct state {
	vector u[3];
	vector ub[3];
	matrix x;
};

std::istream&
operator>>(std::istream& in, state& st)
{
	using namespace util::functional;
	auto load = [&] (auto& v) { in >> linalg::io::binary >> v; };
	map(load, st.u);
	map(load, st.ub);
	load(st.x);
	return in;
}

int
main(int argc, char** argv)
{
	util::set_default_resource(cuda::default_device().memory());
	constexpr bases::polyharmonic_spline<7> basic;

	auto m = 2744;
	if (argc > 1)
		m = atoi(argv[1]);

	state t;
	std::cin >> t;
	if (std::cin.eof()) return -1;

	auto n = t.x.rows();
	rbc ref{n, m, basic};
	bases::container rbcs{ref, std::move(t.x)};
	python_writer write{rbcs};

	while (true) {
		std::cin >> t;
		if (std::cin.eof()) break;
		rbcs.x = std::move(t.x);
		write(rbcs);
	}

	return 0;
}
