#pragma once
#include "types.h"
#include "handle.h"
#include "exceptions.h"

namespace cusolver {

inline void
create(gesvdj_info_t& info)
{
	throw_if_error(cusolverDnCreateGesvdjInfo(&info));
}

inline void
destroy(gesvdj_info_t& info)
{
	throw_if_error(cusolverDnDestroyGesvdjInfo(&info));
}

inline void
set_tolerance(gesvdj_info_t& info, double t)
{
	cusolverDnXgesvdjSetTolerance(info, t);
}

inline void
set_max_sweeps(gesvdj_info_t& info, int s)
{
	cusolverDnXgesvdjSetMaxSweeps(info, s);
}

inline void
set_sort_eigenvalues(gesvdj_info_t& info, bool s)
{
	cusolverDnXgesvdjSetSortEig(info, s);
}

class syevj_info : type_wrapper<gesvdj_info_t> {
private:
	using base = type_wrapper<gesvdj_info_t>;
	using base::data;
public:
	util::cached<double> tolerance;
	util::cached<int> max_sweeps;
	util::cached<bool> sort;

	gesvdj_info() :
		base(),
		tolerance([&] (const double& t) { set_tolerance(data, t); }, 0),
		max_sweeps([&] (const int& s) { set_max_sweeps(data, s); }, 100),
		sort([&] (const bool& s) { set_sort_eigenvalues(data, s); }, true) {}
	gesvdj_info(gesvdj_info_t& info) :
		base(info),
		tolerance([&] (const double& t) { set_tolerance(data, t); }, 0),
		max_sweeps([&] (const int& s) { set_max_sweeps(data, s); }, 100),
		sort([&] (const bool& s) { set_sort_eigenvalues(data, s); }, true) {}
	gesvdj_info(gesvdj_info_t&&) : gesvdj_info() {}
};

inline double
residual(dense::handle& h, syevj_info& info)
{
	double res;
	throw_if_error(cusolverDnXgesvdjGetResidual(h, info, &res));
	return res;
}

inline int
sweeps(dense::handle& h, syevj_info& info)
{
	int sweeps;
	throw_if_error(cusolverDnXgesvdjGetSweeps(h, info, &sweeps));
	return sweeps;
}
}
