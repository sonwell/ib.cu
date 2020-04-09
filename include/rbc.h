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
	using vector = bases::vector;
	using base = bases::shapes::sphere;
	static constexpr bases::traits<rbc> traits;
	static constexpr bases::polynomials<0> p;
public:
	static matrix
	shape(const matrix& params)
	{
		static constexpr double radius = 3.91_um;
		using point = std::array<double, 3>;
		auto k = [=] __device__ (auto params) -> point
		{
			auto [t, p] = params;
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z0 = sin(p);
			auto r2 = x*x + y*y;
			auto z = 0.5 * z0 * (0.21 + 2.0 * r2 - 1.12 * r2*r2);
			return {radius * x, radius * y, radius * z};
		};
		return base::shape(params, k);
	}

	static std::filesystem::path
	filename(int n)
	{
		static std::filesystem::path root = "data";
		std::stringstream ss;
		ss << "rbc." << n << ".bin";
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
			util::logging::warn(n, " is not specialized for rbc; ",
					"falling back to sphere sampling.");
			return bases::shapes::sphere::sample(n);
		}

		matrix m;
		auto name = filename(n);
		std::fstream f(name, std::ios::in | std::ios::binary);
		f >> linalg::io::binary >> m;
		return m;
	}

	template <typename interp, typename eval,
			 typename = std::enable_if_t<bases::is_basic_function_v<interp>>,
			 typename = std::enable_if_t<bases::is_basic_function_v<eval>>>
	rbc(int nd, int ns, interp phi, eval psi) :
		bases::shapes::sphere(nd, ns, traits, phi, psi, p) {}

	template <typename basic,
			 typename = std::enable_if_t<bases::is_basic_function_v<basic>>>
	rbc(int nd, int ns, basic phi) :
		rbc(nd, ns, phi, phi) {}
};
