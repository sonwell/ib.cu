#pragma once
#include "rbf.h"
#include "operators.h"
#include "geometry.h"
#include "traits.h"

namespace bases {

// surface base class. Holds geometry data and operators.
template <int dims>
struct surface {
public:
	static constexpr auto dimensions = dims;
private:
	using operators = bases::operators<dimensions>;
	using geometry = bases::geometry<dimensions>;

	typedef struct {
		matrix sites;
		vector weights;
	} partial_info;

	typedef struct {
		partial_info data;
		partial_info sample;
		matrix positions;
	} info;

	template <meta::traits traits>
	static partial_info
	do_sample(int n, traits)
	{
		auto sites = traits::sample(n);
		auto weights = traits::weights(sites);
		return {std::move(sites), std::move(weights)};
	}

	template <meta::traits traits>
	static info
	get_info(int nd, int ns, traits tr)
	{
		auto data = do_sample(nd, tr);
		auto sample = nd == ns ?  data : do_sample(ns, tr);
		auto x = traits::shape(data.sites);
		return {std::move(data), std::move(sample), std::move(x)};
	}

	template <meta::rbf interp, meta::rbf eval, meta::polynomial poly>
	surface(int nd, int ns, info info, interp phi, eval psi, poly p) :
		num_data_sites(nd), num_sample_sites(ns),
		data_to_data(info.data.sites, info.data.sites,
				std::move(info.data.weights), phi, phi, p),
		data_to_sample(info.data.sites, nd == ns ? info.data.sites : info.sample.sites,
				std::move(info.sample.weights), phi, psi, p),
		data_geometry(data_to_data, info.positions),
		sample_geometry(data_to_sample, info.positions) {}

	template <meta::traits traits, meta::rbf interp, meta::rbf eval,
	          meta::polynomial poly>
	surface(int nd, int ns, traits tr, interp phi, eval psi, poly p) :
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
	operators data_to_data;
	operators data_to_sample;
	geometry data_geometry;
	geometry sample_geometry;

	template <meta::traits traits, meta::basic interp, meta::basic eval,
	          meta::metric metric, meta::polynomial poly>
	surface(int nd, int ns, traits tr, interp phi, eval psi, metric d, poly p) :
		surface(nd, ns, tr, rbf{phi, d}, rbf{psi, d}, p) {}

	template <meta::traits traits, meta::basic basic, meta::metric metric,
	          meta::polynomial poly>
	surface(int nd, int ns, traits tr, basic phi, metric d, poly p) :
		surface(nd, ns, tr, phi, phi, d, p) {}
};

template <typename T> struct is_shape : std::is_base_of<surface<T::dimensions>, T> {};
template <typename T> inline constexpr auto is_shape_v = is_shape<T>::value;

namespace meta { template <typename T> concept shape = is_shape_v<T>; }

}
