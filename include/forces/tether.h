#pragma once
#include "bases/geometry.h"
#include "units.h"
#include "types.h"
#include "load.h"

namespace forces {

struct tether {
	using energy_per_area = units::unit<0, 1, -2>;
	energy_per_area modulus;

	template <typename object_type>
	decltype(auto)
	operator()(const object_type& object) const
	{
		using bases::reference;
		using bases::current;

		const auto& orig = object.geometry(reference);
		const auto& curr = object.geometry(current);
		loader original{orig};
		loader deformed{curr};

		auto size = linalg::size(curr.position);
		auto n = size.rows * size.cols / 3;
		matrix f{size};

		auto* fdata = f.values();
		auto k = [=, modulus=modulus] __device__ (int tid)
		{
			auto orig = original[tid];
			auto curr = deformed[tid];
			auto& ox = orig.x;
			auto& cx = curr.x;

			for (int i = 0; i < 3; ++i)
				fdata[i * n + tid] = modulus * (ox[i] - cx[i]);
		};
		util::transform(k, n);
		return f;
	}

	constexpr tether(energy_per_area modulus) :
		modulus(modulus) {}
};

} // namespace force
