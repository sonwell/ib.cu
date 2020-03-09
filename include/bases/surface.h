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

	template <typename traits_type, typename rbf>
	static partial_info
	do_sample(int n, traits<traits_type>, rbf phi)
	{
		auto sites = traits<traits_type>::sample(n);
		auto weights = traits<traits_type>::weights(sites, phi);
		return {std::move(sites), std::move(weights)};
	}

	template <typename traits_type, typename rbf>
	static info
	get_info(int nd, int ns, traits<traits_type> tr, rbf phi)
	{
		auto data = do_sample(nd, tr, phi);
		auto sample = nd == ns ?
			data : do_sample(ns, tr, phi);
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
				std::move(info.data.weights), phi, psi, p),
		data_to_sample(info.data.sites, nd == ns ? info.data.sites : info.sample.sites,
				std::move(info.sample.weights), phi, psi, p),
		data_geometry(data_to_data, info.positions),
		sample_geometry(data_to_sample, info.positions) {}

	template <typename traits_type, typename interp, typename eval, typename poly>
	surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, poly p) :
		surface(nd, ns, get_info(nd, ns, tr, phi), phi, psi, p) {}
public:
	int num_data_sites;
	int num_sample_sites;
	operators_type data_to_data;
	operators_type data_to_sample;
	geometry_type data_geometry;
	geometry_type sample_geometry;

	template <typename traits_type, typename interp, typename eval, typename metric, typename poly,
			 typename = std::enable_if_t<is_basic_function_v<interp> && is_basic_function_v<eval> &&
				 is_metric_v<metric>>>
	surface(int nd, int ns, traits<traits_type> tr, interp phi, eval psi, metric d, poly p) :
		surface(nd, ns, tr, rbf{phi, d}, rbf{psi, d}, p) {}

	template <typename traits_type, typename basic, typename metric, typename poly,
			 typename = std::enable_if_t<is_basic_function_v<basic> && is_metric_v<metric>>>
	surface(int nd, int ns, traits<traits_type> tr, basic phi, metric d, poly p) :
		surface(nd, ns, tr, phi, phi, d, p) {}
};

}
