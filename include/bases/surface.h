#pragma once
#include "rbf.h"
#include "operators.h"
#include "geometry.h"
#include "traits.h"

namespace bases {

template <int dims>
struct surface {
public:
	static constexpr auto dimensions = dims;
private:
	using operators_type = operators<dimensions>;
	using geometry_type = geometry<dimensions>;

	typedef struct {
		matrix sites;
		vector weights;
	} partial_info;

	typedef struct {
		partial_info data;
		partial_info sample;
		matrix positions;
	} info;

	template <typename traits_type>
	static partial_info
	do_sample(int n, traits<traits_type>)
	{
		auto sites = traits<traits_type>::sample(n);
		auto weights = traits<traits_type>::weights(sites);
		return {std::move(sites), std::move(weights)};
	}

	template <typename traits_type>
	static info
	get_info(int nd, int ns, traits<traits_type> tr)
	{
		auto data = do_sample(nd, tr);
		auto sample = nd == ns ?  data : do_sample(ns, tr);
		auto x = traits<traits_type>::shape(data.sites);
		return {
			std::move(data),
			std::move(sample),
			std::move(x),
		};
	}

	template <typename interp, typename eval, typename poly>
	surface(int nd, int ns, info info, interp phi, eval psi, poly p) :
		num_data_sites(nd), num_sample_sites(ns),
		data_to_data(info.data.sites, info.data.sites,
				std::move(info.data.weights), phi, phi, p),
		data_to_sample(info.data.sites, nd == ns ? info.data.sites : info.sample.sites,
				std::move(info.sample.weights), phi, psi, p),
		data_geometry(data_to_data, info.positions),
		sample_geometry(data_to_sample, info.positions) {}

	template <typename traits_type, typename interp, typename eval, typename poly>
	surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, poly p) :
		surface(nd, ns, get_info(nd, ns, tr), phi, psi, p) {}
protected:
	template <typename f_type>
	static matrix
	shape(const matrix& params, f_type&& f)
	{
		auto rows = params.rows();
		matrix x(rows, dimensions+1);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid, auto f)
		{
			std::array<double, dimensions> p;
			for (int i = 0; i < dimensions; ++i)
				p[i] = pdata[i * rows + tid];
			auto x = f(std::move(p));
			for (int i = 0; i < dimensions+1; ++i)
				xdata[i * rows + tid] = x[i];
		};
		util::transform<128, 8>(k, rows, std::forward<f_type>(f));
		return x;
	}
public:
	int num_data_sites;
	int num_sample_sites;
	operators_type data_to_data;
	operators_type data_to_sample;
	geometry_type data_geometry;
	geometry_type sample_geometry;

	template <typename traits_type, typename interp, typename eval, typename metric, typename poly>
	surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, metric d, poly p) :
		surface(nd, ns, tr, rbf{phi, d}, rbf{psi, d}, p) {}

	template <typename traits_type, typename basic, typename metric, typename poly,
	          typename = std::enable_if_t<is_basic_function_v<basic> &&
	                                      is_metric_v<metric> &&
	                                      is_polynomial_basis_v<poly>>>
	surface(int nd, int ns, traits<traits_type> tr, basic phi, metric d, poly p) :
		surface(nd, ns, tr, phi, phi, d, p) {}
};

}
