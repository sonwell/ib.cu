#pragma once
#include "types.h"
#include "handle.h"
#include "exceptions.h"

namespace cusolver {

inline void
create(syevj_info_t& info)
{
	throw_if_error(cusolverDnCreateSyevjInfo(&info));
}

inline void
destroy(syevj_info_t& info)
{
	throw_if_error(cusolverDnDestroySyevjInfo(info));
}

inline void
set_tolerance(syevj_info_t& info, double t)
{
	cusolverDnXsyevjSetTolerance(info, t);
}

inline void
set_max_sweeps(syevj_info_t& info, int s)
{
	cusolverDnXsyevjSetMaxSweeps(info, s);
}

inline void
set_sort_eigenvalues(syevj_info_t& info, bool s)
{
	cusolverDnXsyevjSetSortEig(info, s);
}

struct syevj_info : cusolver::type_wrapper<syevj_info_t> {
private:
	using base = cusolver::type_wrapper<syevj_info_t>;
	using base::value;
public:
	util::cached<double> tolerance;
	util::cached<int> max_sweeps;
	util::cached<bool> sort;

	syevj_info() :
		base(),
		tolerance([&] (const double& t) { set_tolerance(value, t); }, 0),
		max_sweeps([&] (const int& s) { set_max_sweeps(value, s); }, 100),
		sort([&] (const bool& s) { set_sort_eigenvalues(value, s); }, true) {}
	syevj_info(syevj_info_t& info) :
		base(info),
		tolerance([&] (const double& t) { set_tolerance(value, t); }, 0),
		max_sweeps([&] (const int& s) { set_max_sweeps(value, s); }, 100),
		sort([&] (const bool& s) { set_sort_eigenvalues(value, s); }, true) {}
	syevj_info(syevj_info_t&& info) : syevj_info() {}
};

inline double
residual(dense::handle& h, syevj_info& info)
{
	double res;
	throw_if_error(cusolverDnXsyevjGetResidual(h, info, &res));
	return res;
}

inline int
sweeps(dense::handle& h, syevj_info& info)
{
	int sweeps;
	throw_if_error(cusolverDnXsyevjGetSweeps(h, info, &sweeps));
	return sweeps;
}

}
