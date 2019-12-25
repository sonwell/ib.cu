#pragma once
#include "rbf.h"
#include "operators.h"
#include "geometry.h"

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

	template <typename traits, typename rbf>
	static partial_info
	do_sample(int n, traits, rbf phi)
	{
		auto sites = traits::sample(n);
		auto weights = traits::weights(sites, phi);
		return {std::move(sites), std::move(weights)};
	}

	template <typename traits, typename rbf>
	static info
	get_info(int nd, int ns, traits tr, rbf phi)
	{
		auto data = do_sample(nd, tr, phi);
		auto sample = nd == ns ?
			data : do_sample(ns, tr, phi);
		auto x = traits::shape(data.sites);
		return {
			std::move(data),
			std::move(sample),
			std::move(x),
		};
	}

	template <typename rbf, typename poly>
	surface(int nd, int ns, info info, rbf phi, poly p) :
		num_data_sites(nd), num_sample_sites(ns),
		data_to_data(info.data.sites, info.data.sites,
				std::move(info.data.weights), phi, p),
		data_to_sample(info.data.sites, nd == ns ? info.data.sites : info.sample.sites,
				std::move(info.sample.weights), phi, p),
		data_geometry(data_to_data, info.positions),
		sample_geometry(data_to_sample, info.positions) {}

	template <typename traits, typename rbf, typename poly>
	surface(int nd, int ns, traits tr, rbf phi, poly p) :
		surface(nd, ns, get_info(nd, ns, tr, phi), phi, p) {}
public:
	using params = double[dimensions];

	int num_data_sites;
	int num_sample_sites;
	operators_type data_to_data;
	operators_type data_to_sample;
	geometry_type data_geometry;
	geometry_type sample_geometry;

	template <typename traits, typename basic, typename metric, typename poly>
	surface(int nd, int ns, traits tr, basic phi, metric d, poly p) :
		surface(nd, ns, tr, bases::rbf{phi, d}, p) {}
};

}
