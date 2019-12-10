#pragma once
#include "bases/geometry.h"
#include "types.h"
#include "load.h"
#include "units.h"

namespace forces {

struct repelling {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area modulus;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		constexpr auto height = 16_um;
		constexpr auto h = 0.25_um;

		using bases::current;
		const auto& curr = object.geometry(current).sample;
		loader deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size, linalg::zero};

		auto* fdata = f.values();
		auto k = [=, modulus=modulus] __device__ (int tid)
		{
			auto curr = deformed(tid);
			auto& cx = curr.x;
			auto r = abs(height - cx[1]) / h;

			auto fy = r < 2 ?  -log(r / 2) * exp(-4/(4-r*r)) : 0.0;
			double f[3] = {0.0, fy, 0.0};
			for (int i = 0; i < 3; ++i)
				fdata[i * n + tid] = -modulus * f[i] * curr.s;
		};
		util::transform(k, n);
		return f;
	}

	constexpr repelling(energy_per_area modulus) :
		modulus(modulus) {}
};

} // namespace force
