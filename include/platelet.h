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
	static constexpr double major = 1.55_um;
	static constexpr double minor = 0.50_um;
public:
	static matrix
	shape(const matrix& params)
	{
		using point = std::array<double, 3>;
		auto k = [=] __device__ (auto params) -> point
		{
			auto [t, p] = params;
			auto x = cos(t) * cos(p);
			auto y = sin(t) * cos(p);
			auto z = sin(p);
			return {major * x, minor * y, major * z};
		};
		return base::shape(params, k);
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

	template <bases::meta::basic interp, bases::meta::basic eval>
	platelet(int nd, int ns, interp phi, eval psi) :
		bases::shapes::sphere(nd, ns, traits, phi, psi, p) {}

	template <bases::meta::basic basic>
	platelet(int nd, int ns, basic phi) :
		platelet(nd, ns, phi, phi) {}
};

struct platelet1d : bases::shapes::circle {
private:
	using matrix = bases::matrix;
	using vector = bases::vector;
	using base = bases::shapes::circle;
	static constexpr bases::traits<platelet1d> traits;
	static constexpr bases::polynomials<0> p;
	static constexpr double major = 1.82_um;
	static constexpr double minor = 0.46_um;
protected:
	using base::shape;
public:
	static matrix
	shape(const matrix& params)
	{
		using point = std::array<double, 2>;
		auto k = [=] __device__ (auto p) -> point
		{
			auto [t] = p;
			return {major * cos(t), minor * sin(t)};
		};
		return base::shape(params, k);
	}

	template <bases::meta::basic interp, bases::meta::basic eval>
	platelet1d(int nd, int ns, interp phi, eval psi) :
		bases::shapes::circle(nd, ns, traits, phi, psi, p) {}

	template <bases::meta::basic basic>
	platelet1d(int nd, int ns, basic phi) :
		platelet1d(nd, ns, phi, phi) {}
};
