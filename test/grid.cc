#include "util/sequences.h"
#include "util/array.h"
#include "util/launch.h"
#include "units.h"
#include "fd/grid.h"
#include "fd/domain.h"
#include "fd/dimension.h"
#include "fd/boundary.h"

template <std::size_t n, typename domain_type>
struct grid {
public:
	static constexpr auto dimensions = domain_type::ndim;
private:
	template <typename collection, typename getter, std::size_t ... ns>
	static constexpr auto
	collect(const collection& c, getter&& get, util::sequence<std::size_t, ns...>)
	{
		return util::array{get(std::get<ns>(c))...};
	}

	template <typename collection, typename getter>
	static constexpr auto
	collect(const collection& c, getter&& get)
	{
		constexpr auto size = std::tuple_size_v<collection>;
		using sequence = util::make_sequence<std::size_t, size>;
		return collect(c, std::forward<getter>(get), sequence{});
	}

	using tag_type = typename domain_type::tag_type;
	static constexpr auto ggp = [] (auto&& d) { return d.grid_points(); };
public:
	using grid_type = fd::grid::make<tag_type, n, dimensions>;

	util::array<double, dimensions> shifts;
	util::array<bool, dimensions> on_boundary;
	util::array<bool, dimensions> solid_boundary;
	util::array<int, dimensions> grid_cells;

	constexpr grid(const domain_type& domain) :
		shifts(grid_type::shifts),
		on_boundary(grid_type::on_boundary),
		solid_boundary(domain_type::solid_boundary),
		grid_cells(collect(fd::dimensions(domain), ggp)) {}
};

int
main(void)
{
	constexpr fd::dimension x(1_m, fd::boundary::periodic());
	constexpr fd::dimension y(1_m, fd::boundary::dirichlet());
	constexpr fd::domain domain(fd::grid::mac(16), x, y);
	using grid_type = grid<0, decltype(domain)>;
	grid_type g(domain);

	auto k = [=] __device__ (int tid)
	{

		printf("%d %f %d %d %d\n", tid, g.shifts[tid], g.grid_cells[tid],
			g.on_boundary[tid], g.solid_boundary[tid]);
	};
	util::transform(k, domain.ndim);
	cudaDeviceSynchronize();

	return 0;
}
