#pragma once
#include <fstream>
#include <filesystem>
#include "util/log.h"
#include "bases/types.h"
#include "bases/shapes/sphere.h"
#include "bases/traits.h"
#include "bases/polynomials.h"
#include "units.h"

struct platelet : bases::shapes::sphere {
private:
	using matrix = bases::matrix;
	using vector = bases::vector;
	using base = bases::shapes::sphere;
	static constexpr bases::traits<platelet> traits;
	static constexpr bases::polynomials<0> p;
public:
	static matrix
	shape(const matrix& params)
	{
		static constexpr double major = 1.82_um;
		static constexpr double minor = 0.46_um;
		auto rows = params.rows();
		matrix x(rows, 3);

		auto* pdata = params.values();
		auto* xdata = x.values();
		auto k = [=] __device__ (int tid)
		{
			auto t = pdata[0 * rows + tid];
			auto p = pdata[1 * rows + tid];
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z = sin(p);

			xdata[0 * rows + tid] = major * x;
			xdata[1 * rows + tid] = minor * y;
			xdata[2 * rows + tid] = major * z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	static std::filesystem::path
	filename(int n)
	{
		static std::filesystem::path root = "data";
		std::stringstream ss;
		ss << "plt." << n << ".bin";
		return root / ss.str();
	}

	static bool
	is_specialized(int n)
	{
		return std::filesystem::exists(filename(n));
	}

	static matrix
	sample(int n)
	{
		if (!is_specialized(n)) {
			util::logging::warn(n, " is not specialized for platelet; ",
					"falling back to sphere sampling.");
			return bases::shapes::sphere::sample(n);
		}

		matrix m;
		auto name = filename(n);
		std::fstream f(name, std::ios::in | std::ios::binary);
		f >> linalg::io::binary >> m;
		return m;
	}

	template <typename rbf>
	static vector
	weights(const matrix& x, rbf phi)
	{
		if (!is_specialized(x.rows()))
			return base::weights(x, phi);
		static constexpr double surface_area = 13.68084640609756467093;
		static constexpr auto weight = [] __device__ (const params& x)
		{
			auto cphi = cos(x[1]);
			auto r2 = cphi * cphi;
			auto scale = 1.82 * cphi * sqrt(0.46 * 0.46 + (1.82 * 1.82 - 0.46 * 0.46) * r2);
			return scale / surface_area;
		};
		return base::weights(x, phi, weight);
	}

	template <typename basic>
	platelet(int nd, int ns, basic phi) :
		bases::shapes::sphere(nd, ns, traits, phi, p) {}
};
