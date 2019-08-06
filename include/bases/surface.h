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
		matrix data_sites;
		matrix sample_sites;
		matrix positions;
		vector data_weights;
		vector sample_weights;
	} sample_info;

	template <typename traits, typename rbf>
	static sample_info
	do_sample(int nd, int ns, traits, rbf phi)
	{
		auto data_sites = traits::sample(nd);
		auto sample_sites = traits::sample(ns);
		auto positions = traits::shape(data_sites);
		auto data_weights = traits::weights(data_sites, phi);
		auto sample_weights = traits::weights(sample_sites, phi);
		return {
			std::move(data_sites),
			std::move(sample_sites),
			std::move(positions),
			std::move(data_weights),
			std::move(sample_weights)
		};
	}

	template <typename rbf, typename poly>
	surface(int nd, int ns, sample_info info, rbf phi, poly p) :
		num_data_sites(nd), num_sample_sites(ns),
		data_to_data(info.data_sites, info.data_sites, info.data_weights, phi, p),
		data_to_sample(info.data_sites, info.sample_sites, info.sample_weights, phi, p),
		data_geometry(data_to_data, info.positions),
		sample_geometry(data_to_sample, info.positions) {}

	template <typename traits, typename rbf, typename poly>
	surface(int nd, int ns, traits tr, rbf phi, poly p) :
		surface(nd, ns, do_sample(nd, ns, tr, phi), phi, p) {}
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
