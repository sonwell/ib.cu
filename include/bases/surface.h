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
	using operators_type = bases::operators<dimensions>;
	using geometry_type = bases::geometry<dimensions>;

	typedef struct sample {
		matrix reduced;
		matrix sites;
		vector weights;
		matrix positions;

		template <meta::traits traits>
		sample(int n, traits) :
			reduced(traits::sample(traits::reduce(n))),
			sites(traits::sample(n)),
			weights(traits::weights(sites)),
			positions(traits::shape(sites)) {}
	} sample;

	template <meta::rbf interp, meta::rbf eval, meta::polynomial poly>
	surface(sample sample, interp phi, eval psi, poly p) :
		samples(sample.sites.rows()),
		operators(sample.reduced, sample.sites,
				std::move(sample.weights), phi, phi, p),
		geometry(operators, sample.positions) {}

	template <meta::traits traits, meta::rbf interp, meta::rbf eval,
	          meta::polynomial poly>
	surface(int n, traits tr, interp phi, eval psi, poly p) :
		surface(sample{n, tr}, phi, psi, p) {}
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
	int samples;
	operators_type operators;
	geometry_type geometry;

	template <meta::traits traits, meta::basic interp, meta::basic eval,
	          meta::metric metric, meta::polynomial poly>
	surface(int n, traits tr, interp phi, eval psi, metric d, poly p) :
		surface(n, tr, rbf{phi, d}, rbf{psi, d}, p) {}

	template <meta::traits traits, meta::basic basic, meta::metric metric,
	          meta::polynomial poly>
	surface(int n, traits tr, basic phi, metric d, poly p) :
		surface(n, tr, phi, phi, d, p) {}
};

template <typename T> struct is_shape : std::is_base_of<surface<T::dimensions>, T> {};
template <typename T> inline constexpr auto is_shape_v = is_shape<T>::value;

namespace meta { template <typename T> concept shape = is_shape_v<T>; }

}
