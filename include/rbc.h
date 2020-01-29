#pragma once
#include <fstream>
#include <filesystem>
#include "util/log.h"
#include "bases/types.h"
#include "bases/shapes/sphere.h"
#include "bases/traits.h"
#include "bases/polynomials.h"
#include "units.h"

struct rbc : bases::shapes::sphere {
private:
	using matrix = bases::matrix;
	static constexpr bases::traits<rbc> traits;
	static constexpr bases::polynomials<0> p;
public:
	static matrix
	shape(const matrix& params)
	{
		static constexpr double radius = 3.91_um;
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
			auto z0 = sin(p);
			auto r2 = x*x + y*y;
			auto z = 0.5 * z0 * (0.21 + 2.0 * r2 - 1.12 * r2*r2);

			xdata[0 * rows + tid] = radius * x;
			xdata[1 * rows + tid] = radius * y;
			xdata[2 * rows + tid] = radius * z;
		};
		util::transform<128, 8>(k, rows);
		return x;
	}

	static matrix
	sample(int n)
	{
		static std::filesystem::path root = "data";
		std::fstream f;

		std::filesystem::path file = root;
		std::stringstream ss;
		ss << "rbc." << n << ".bin";
		file /= ss.str();

		if (!std::filesystem::exists(file)) {
			util::logging::warn(n, " is not specialized for rbc; ",
					"falling back to sphere sampling.");
			return bases::shapes::sphere::sample(n);
		}

		matrix m;
		f.open(file, std::ios::in | std::ios::binary);
		f >> linalg::io::binary >> m;
		return m;
	}

	template <typename basic>
	rbc(int nd, int ns, basic phi) :
		bases::shapes::sphere(nd, ns, traits, phi, p) {}
};
